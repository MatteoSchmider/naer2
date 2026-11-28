"""
Recovery: every runtime's launches on one device, kept until known complete, and what brings them back when a
run stops -- one per device (its runtimes read each other's emission histories), recovery(device).

A run whose ring is too small for it sets the stop word ctl the kernels share and stops without committing
anything (fused.py: _pull); every run that starts after it stops at once (_run_gate) -- this stream's and every
other's. recover() brings them all back: once no kernel runs any more, every ring that stopped a run grows to
what it needs, and every launch not known complete runs again, in launch order (its groups that finished
skip), the device idle again after. Nothing is lost, and the result is what it would have been with the
larger ring from the start.

resize() applies the sizes the runtimes' sizers want (naer2/sim/sizing.py), together, once the device is idle.
"""

import collections
import weakref

import torch


class Window:
    """A replayed window's runs as one journal entry: per runtime its part (naer2/sim/ledger.py: WindowPart),
    and the order its runs were launched in (round by round) -- a recovery reruns the parts not yet judged
    complete, in that order."""

    def __init__(self, order):
        self._order = order  # () -> [(runtime, entry index, run id)], round by round (asked for on a rerun only)
        self.parts: dict = {}  # runtime id -> WindowPart, until judged complete

    def runs(self):
        """(runtime, RunParams) of every run of a part still pending, in launch order."""
        for rt, i, run_id in self._order():
            part = self.parts.get(id(rt))
            if part is not None:
                yield rt, rt.entry_params(part.sched, i, run_id, status=part.status)

    def done(self, rt) -> bool:
        """rt's part judged complete; whether the whole window is."""
        self.parts.pop(id(rt), None)
        return not self.parts


class Recovery:
    """See the module docstring. journal: (runtime id, run id) -> (runtime, RunParams) of a direct or planned
    run; ("window", id) -> a replayed window's runs (Window) -- in launch order."""

    def __init__(self, device):
        self.ctl = torch.zeros(1, device=device, dtype=torch.int32)
        self.journal = collections.OrderedDict()
        self.runtimes = weakref.WeakSet()
        self.recoveries = 0
        self._resizing = False

    def launched(self, rt, params) -> None:
        self.journal[(id(rt), params.run_id)] = (rt, params)

    def launched_window(self, window: Window) -> None:
        self.journal[("window", id(window))] = window

    def complete(self, rt, run_id: int) -> None:
        self.journal.pop((id(rt), run_id), None)

    def window_complete(self, window: Window, rt) -> None:
        if window.done(rt):
            self.journal.pop(("window", id(window)), None)

    def resize(self) -> None:
        """Every resize a runtime's sizer wants, together: once no kernel runs on the device any more (a
        precaution: reallocating while other streams' kernels ran faulted the GPU, see TODO.md) -- one drain
        for all of them, every runtime's landed statuses judged first (any resize they want joins in)."""
        if self._resizing:
            return  # (a resize or recovery under way applies it)
        self._resizing = True
        try:
            torch.cuda.synchronize()
            for rt in list(self.runtimes):
                rt.ledger.judge_pending(wait=True)
            for rt in list(self.runtimes):
                rt.sizer.apply(rt.ring, rt.run_id)
            # and once the resizes' copies are done: a resize runs while a round is being launched (a run's
            # judging asks for it), and the round's populations' streams wait only for what the current stream
            # held when the round began -- their kernels would read the new rings before the copies filled
            # them (PD14's volley: free stacks of zeros, two slots in one block, an illegal address)
            torch.cuda.synchronize()
        finally:
            self._resizing = False

    def recover(self) -> None:
        resizing, self._resizing = self._resizing, True
        try:
            self._recover()
        finally:
            self._resizing = resizing

    def _recover(self) -> None:
        while True:
            torch.cuda.synchronize()  # nothing runs from here on: the rings can be reallocated
            for rt in list(self.runtimes):
                rt.ledger.clear()  # (stale: their runs stopped, or are rerun below)
            for rt in list(self.runtimes):
                rt.sizer.stopped(rt.ring, rt.run_id)
                rt.sizer.apply(rt.ring, rt.run_id)
            self.ctl.zero_()
            for entry in list(self.journal.values()):
                for rt, params in (entry.runs() if isinstance(entry, Window) else (entry,)):
                    rt.launch(params, None)
            torch.cuda.synchronize()
            self.recoveries += 1
            if not int(self.ctl[0]):
                break
        self.journal.clear()

    def settle(self) -> None:
        """Every runtime's pending statuses judged (waiting for them): a stopped run is rerun before the host
        reads what the runs wrote (a history, a ring) -- every host readback of device state goes through here."""
        for rt in list(self.runtimes):
            rt.ledger.judge_pending(wait=True)


_RECOVERY: dict = {}


def recovery(device) -> Recovery:
    """The recovery of every runtime on this device (one stop word, one journal)."""
    device = torch.device(device)
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device()
           if device.type == "cuda" else 0)
    if key not in _RECOVERY:
        _RECOVERY[key] = Recovery(device)
    return _RECOVERY[key]
