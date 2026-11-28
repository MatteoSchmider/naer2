"""
Running what the scheduler decides (naer2/sim/scheduler.py), one network's: a round's runs launched -- one fused
kernel where the round's runtimes are exactly the fused round's (naer2/fusion.py), else each population's own
kernel on its own stream, joined -- three ways:

  step(runs, round_id)         a stepped round: each run's entry in its runtime's direct schedule (pinned host
                               memory), the runs launched now (PopulationRuntime.direct_params)
  planned(launches, round0)    a planned window: every run's entry in device memory (naer2/schedule.py: one
                               upload), every round launched from it
  captured(launches, ...)      a captured window: the window's launches recorded once per shape as a device
                               graph (naer2/graph.py), every window of that shape a replay -- its entries
                               written first; a clock-only network's windows extrapolated (naer2/periodic.py)

Every launch the same bookkeeping (launch_round): its status slot judged free, the run pending and journaled
until known complete (naer2/sim/ledger.py, recovery.py) -- a captured window's runs as one part per runtime
after each replay instead. A window's runs take status slots for two windows (one in flight while the next is
filled); after it, the runtimes move on past it and an event-driven population's ring position comes back from
the device (only its gated kernels know which ran).
"""

import torch

from ..periodic import Periodic
from ..schedule import Schedule


class Executor:
    """See the module docstring. fuse: fused rounds (Network.fuse_rounds); barrier: a captured window's barrier
    kernel between rounds (naer2/graph.py)."""

    def __init__(self, network):
        self.network = network
        self.device = network.device
        self.fuse = True
        self.barrier = False
        self.graphs: dict = {}  # window signature -> WindowGraph
        self.captures = 0
        self.periodic = Periodic()
        self.fused = None
        self._schedule = None  # (a planned window's entries: alive until the next window's)
        self._streams: dict = {}
        self._done: dict = {}
        self._ready = None

    # ---- a round

    def fused_round(self):
        """The fused kernel of every population (None: fusing off, not on CUDA, or a population without its
        runtime): built once, again when a runtime or its kernel is replaced."""
        pops = self.network.populations
        if not self.fuse or self.device.type != "cuda" or any(pop._runtime is None for pop in pops):
            return None
        # a fixed order (the same fused kernel every process -- its compile, its cache): the largest first
        runtimes = sorted((pop._runtime for pop in pops), key=lambda rt: (-rt.n_groups, -rt.n, id(rt)))
        key = tuple((id(rt), id(rt.kernel)) for rt in runtimes)
        if self.fused is None or self.fused.key != key:
            from ..fusion import FusedRound
            self.fused = FusedRound(runtimes)
        return self.fused

    def launch_round(self, runs: list, fused=None, bookkeeping: bool = True) -> None:
        """One round's runs, [(runtime, RunParams)]: one fused launch if fused covers exactly their runtimes,
        else each runtime's own kernel on its stream (concurrently). bookkeeping False: the launches alone (a
        graph capture: the bookkeeping follows each replay)."""
        if fused is not None and fused.covers([rt for rt, _ in runs]):
            by_rt = {id(rt): p for rt, p in runs}
            params = [by_rt[id(rt)] for rt in fused.runtimes]
            if bookkeeping:
                for rt in fused.runtimes:
                    rt.ledger.free_slot()
            fused.launch(params, torch.cuda.current_stream())
            if bookkeeping:
                for rt, p in zip(fused.runtimes, params):
                    rt.ledger.launched(p)
            return
        self.concurrently((rt, lambda stream, rt=rt, p=p: rt.start(p, stream) if bookkeeping else rt.launch(p, stream))
                          for rt, p in runs)

    def concurrently(self, work) -> None:
        """Launch each runtime's work on its own CUDA stream: every stream after what came before (one pre-made
        event recorded on the current stream), the current stream after all of them (one pre-made event per
        stream) -- a device-side join, no host sync. work: (key, fn(stream)); fn(None) runs on the current
        stream (CPU: in order)."""
        if self.device.type != "cuda":
            for _, fn in work:
                fn(None)
            return
        main = torch.cuda.current_stream()
        if self._ready is None:
            self._ready = torch.cuda.Event()
        self._ready.record(main)
        launched = []
        for key, fn in work:
            stream = self._streams.get(key)
            if stream is None:
                stream = self._streams[key] = torch.cuda.Stream()
                self._done[key] = torch.cuda.Event()
            stream.wait_event(self._ready)
            fn(stream)
            self._done[key].record(stream)
            launched.append(key)
        for key in launched:
            main.wait_event(self._done[key])

    # ---- the three ways

    def step(self, runs: list, round_id: int) -> None:
        """A stepped round: Scheduler.take's runs [(population, t_start, t_end, gated)], each from its runtime's
        direct schedule (its runtime built and connected first), launched per population (the stepped path
        unfused: its first rounds compile every population's own kernel, not the fused one)."""
        launch = []
        for pop, t_start, t_end, _ in runs:
            rt = pop._prepared_runtime()
            launch.append((rt, rt.direct_params(t_start, t_end, round_id=round_id)))
        self.launch_round(launch)

    def planned(self, launches: list, round0: int) -> int:
        """A planned window (Network.plan's launches, from round round0 on) launched round by round from its
        entries in device memory. Returns the rounds made."""
        schedule = Schedule(self.network, launches, self.device, round0=round0)
        self._slots(schedule, room=False)
        fused = self.fused_round()
        for runs in schedule.launches:
            self.launch_round([(pop._runtime, pop._runtime.entry_params(schedule.entries(pop._runtime), i, run_id))
                               for pop, i, run_id in runs], fused)
        self._done_window(schedule)
        self._schedule = schedule
        return len(launches)

    def captured(self, max_launches: int, until: float | None, periodic: bool) -> int:
        """A captured window: extrapolated (periodic, where learned) or planned (Network.plan), then replayed
        from its shape's graph (captured first if there is none, or it no longer fits). Returns the rounds
        made."""
        from ..graph import WindowGraph

        net = self.network
        periodic = periodic and until is None
        schedule = self.periodic.next(net, max_launches) if periodic else None
        if schedule is None:
            start = (net.t, net.round_id)
            launches = net.plan(max_launches, until, commit=True)
            if not launches:
                return 0
            schedule = Schedule(net, launches, self.device, round0=start[1], upload=False)
            n_rounds = len(launches)
        else:
            start, launches, n_rounds = None, None, schedule.base.n
        self._slots(schedule, room=True)
        sig = schedule.signature()
        graph = self.graphs.get(sig)
        if graph is None or not graph.valid(schedule):
            graph = self.graphs[sig] = WindowGraph(self, schedule, barrier=self.barrier, fused=self.fused_round())
            self.captures += 1
        graph.replay(schedule)
        self._done_window(schedule)
        if launches is not None:  # (planned: a sample for the periodic windows, where they apply)
            if periodic and Periodic.applies(net, launches):
                self.periodic.learn(net, schedule, start)
            else:
                self.periodic.forget()
        return n_rounds

    # ---- a window's before and after

    @staticmethod
    def _slots(schedule, room: bool) -> None:
        """Status slots for two windows' runs (one window in flight while the next launches); room (a captured
        window: its runs' statuses judged together, after the replay): the older runs judged until this
        window's fit, and the window two turns back judged -- its capture turn's buffers are this window's
        next (status slots rounded up could otherwise leave it pending, its entries overwritten)."""
        for key, rt in schedule.runtimes.items():
            n = schedule.n_runs.get(key, 0)
            rt.ledger.ensure(2 * n)
            if room:
                rt.ledger.make_room(n)
                rt.ledger.judge_windows(keep=1)

    def _done_window(self, schedule) -> None:
        """After a window's launches: the runtimes moved on past it (run ids, ring positions); an event-driven
        population's ring position from the device (its gated runs': only their kernels know which ran)."""
        schedule.commit()
        gated = [pop for pop in self.network.populations if pop._event_driven and pop.incoming]
        if gated:
            torch.cuda.synchronize()
            for pop in gated:
                pop._runtime.sync_ring_pos()
