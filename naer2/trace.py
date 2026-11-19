"""
In-kernel tracing, for development: external profilers don't work alongside
Triton here (rocprofv3 crashes), so kernels stamp themselves when compiled
with a TRACE constexpr (and compile it out without). AMD (RDNA2 / RDNA3+)
and NVIDIA: the stamps are inline assembly per backend (_ASM).

Per program, a header -- wall-clock start and end (10 ns ticks: AMD's 100
MHz REALTIME counter, NVIDIA's %globaltimer / 10; comparable across programs
and kernels), event count -- then (tag << 24 | aux, cycle) events; cycles
are 20 bit (AMD: SHADER_CYCLES; NVIDIA: %clock, masked), decoded as deltas
mod 2^20. A stamp taken after pin(x) / pin_s(x) is after x is computed or
loaded. Tags are each kernel's own (decode() takes their names).

    trace = Trace(n_programs, cap)              # host
    t = trace_begin(trace_ptr, pid, CAP)        # kernel (TRACE)
    n = mark(t, n, TAG, aux, CAP)
    trace_end(t, n)
    trace.decode(("start", ...))                # host: per program dict
"""

import torch
import triton
import triton.language as tl

TRACE_WORDS = 6  # header words: realtime start (int64), realtime end (int64), event count, pad
TRACE_HDR = tl.constexpr(TRACE_WORDS)


def _asm() -> dict:
    """The stamps' inline assembly for this process's backend: cycles, realtime (10 ns ticks), pin (a
    32-bit identity move), pin_s (one moved into a uniform register), each (asm, constraints)."""
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else ""
        # REALTIME: s_memrealtime on RDNA2 (gfx10), s_sendmsg_rtn_b64 on RDNA3+ (gfx11 dropped s_memrealtime)
        rt = ("s_memrealtime $0\ns_waitcnt lgkmcnt(0)" if arch.startswith("gfx10")
              else "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_waitcnt lgkmcnt(0)")
        return dict(cycles=("s_getreg_b32 $0, hwreg(HW_REG_SHADER_CYCLES, 0, 20)", "=s"), realtime=(rt, "=s"),
                    pin=("v_mov_b32 $0, $1", "=v,v"), pin_s=("v_readfirstlane_b32 $0, $1", "=s,v"))
    return dict(cycles=("mov.u32 $0, %clock;", "=r"), realtime=("mov.u64 $0, %globaltimer;", "=l"),
                pin=("mov.b32 $0, $1;", "=r,r"), pin_s=("mov.b32 $0, $1;", "=r,r"))


_ASM = _asm()
CYCLES_ASM, CYCLES_C = tl.constexpr(_ASM["cycles"][0]), tl.constexpr(_ASM["cycles"][1])
REALTIME_ASM, REALTIME_C = tl.constexpr(_ASM["realtime"][0]), tl.constexpr(_ASM["realtime"][1])
PIN_ASM, PIN_C = tl.constexpr(_ASM["pin"][0]), tl.constexpr(_ASM["pin"][1])
PIN_S_ASM, PIN_S_C = tl.constexpr(_ASM["pin_s"][0]), tl.constexpr(_ASM["pin_s"][1])
NV = tl.constexpr(not torch.version.hip)  # NVIDIA: %clock is 32 bit, %globaltimer in ns


@triton.jit
def cycles():
    c = tl.inline_asm_elementwise(CYCLES_ASM, CYCLES_C, [], dtype=tl.int32, is_pure=False, pack=1)
    if NV:
        c = c & 0xFFFFF
    return c


@triton.jit
def realtime():
    t = tl.inline_asm_elementwise(REALTIME_ASM, REALTIME_C, [], dtype=tl.int64, is_pure=False, pack=1)
    if NV:
        t = t // 10
    return t


@triton.jit
def pin(x):
    """x, but only once it is ready (an opaque identity move; 32-bit values)."""
    w = x.to(tl.int32, bitcast=True)
    w = tl.inline_asm_elementwise(PIN_ASM, PIN_C, [w], dtype=tl.int32, is_pure=False, pack=1)
    return w.to(x.dtype, bitcast=True)


@triton.jit
def pin_s(x):
    """pin for a (uniform) 32-bit scalar, kept in a uniform register (AMD: an SGPR) so nothing
    downstream turns divergent."""
    w = x.to(tl.int32, bitcast=True)
    w = tl.inline_asm_elementwise(PIN_S_ASM, PIN_S_C, [w], dtype=tl.int32, is_pure=False, pack=1)
    return w.to(x.dtype, bitcast=True)


@triton.jit
def mark(trace, n_ev, tag, aux, TRACE_CAP: tl.constexpr):
    t = cycles()
    ok = n_ev < TRACE_CAP
    tl.store(trace + TRACE_HDR + 2 * n_ev, (tag << 24) | tl.minimum(aux, 0xFFFFFF), mask=ok)
    tl.store(trace + TRACE_HDR + 2 * n_ev + 1, t, mask=ok)
    return n_ev + 1


@triton.jit
def trace_begin(trace_ptr, pid, TRACE_CAP: tl.constexpr):
    """This program's trace record, its wall-clock start stamped."""
    trace = trace_ptr + pid * (TRACE_HDR + 2 * TRACE_CAP)
    tl.store(trace.to(tl.pointer_type(tl.int64)), realtime())
    return trace


@triton.jit
def trace_end(trace, n_ev):
    tl.store(trace + 4, n_ev)
    tl.store((trace + 2).to(tl.pointer_type(tl.int64)), realtime())


class Trace:
    """A trace buffer for n_programs programs of up to cap events each."""

    def __init__(self, n_programs: int, cap: int, device="cuda", level: int = 1):
        self.n_programs, self.cap, self.level = n_programs, cap, level
        self.buf = torch.zeros(n_programs * (TRACE_WORDS + 2 * cap), device=device, dtype=torch.int32)

    def decode(self, tags) -> list[dict]:
        """Per program: wall-clock start / end (10 ns ticks; 0 if the
        program didn't trace), events as (tag, aux, cycles since its first
        event, unwrapped), and how many events didn't fit."""
        tr = self.buf.view(self.n_programs, TRACE_WORDS + 2 * self.cap).cpu()
        rt = tr[:, :4].contiguous().view(torch.int64)
        out = []
        for g in range(self.n_programs):
            n = min(int(tr[g, 4]), self.cap)
            events, prev, acc = [], None, 0
            for word, cyc in tr[g, TRACE_WORDS:TRACE_WORDS + 2 * n].view(n, 2).tolist():
                if prev is not None:
                    acc += (cyc - prev) % (1 << 20)
                prev = cyc
                events.append((tags[(word >> 24) & 0xFF], word & 0xFFFFFF, acc))
            out.append(dict(start=int(rt[g, 0]), end=int(rt[g, 1]), events=events, dropped=int(tr[g, 4]) - n))
        return out
