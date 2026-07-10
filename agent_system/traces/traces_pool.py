"""
Traces Pool: streaming trajectory collection with background compression.

Compression pipeline (applied per episode as it arrives):
  1. State diff   – only record observation deltas, skip unchanged obs
  2. Prefix tree  – merge common action prefixes across trajectories
  3. Loop filter  – remove repeated (action, obs) cycles with no progress
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Step:
    obs: str
    action: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    task_type: str
    task: str
    steps: list[Step]
    outcome: str          # "success" | "failure" | "timeout"
    skill_ids: list[str]  # skills injected during this episode

    # set after compression
    compressed_steps: Optional[list[Step]] = None

    @property
    def effective_steps(self) -> list[Step]:
        return self.compressed_steps if self.compressed_steps is not None else self.steps


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _diff_compress(steps: list[Step]) -> list[Step]:
    """Keep only steps where the observation changed."""
    out = []
    prev_obs = None
    for s in steps:
        if s.obs != prev_obs:
            out.append(s)
            prev_obs = s.obs
        elif s.done or s.reward != 0:
            # Always keep terminal / rewarded steps even if obs unchanged
            out.append(s)
    return out or steps[:1]


def _loop_filter(steps: list[Step], window: int = 6) -> list[Step]:
    """Remove repeated (action, obs) cycles that produce no reward."""
    seen: dict[tuple, int] = {}  # (action, obs) -> last index kept
    out = []
    for i, s in enumerate(steps):
        key = (s.action, s.obs)
        if key in seen and s.reward == 0 and not s.done:
            # duplicate with no progress – skip
            continue
        seen[key] = i
        out.append(s)
    return out or steps[:1]


class _PrefixNode:
    __slots__ = ('children', 'traces')

    def __init__(self):
        self.children: dict[str, '_PrefixNode'] = {}
        self.traces: list[int] = []  # trace indices that pass through here


def _build_prefix_tree(traces: list[Trace]) -> _PrefixNode:
    root = _PrefixNode()
    for idx, trace in enumerate(traces):
        node = root
        for step in trace.effective_steps:
            key = step.action
            if key not in node.children:
                node.children[key] = _PrefixNode()
            node = node.children[key]
            node.traces.append(idx)
    return root


def _shared_prefix_length(traces: list[Trace]) -> int:
    """Return the length of the longest common action prefix across all traces."""
    if not traces:
        return 0
    min_len = min(len(t.effective_steps) for t in traces)
    for i in range(min_len):
        actions = {t.effective_steps[i].action for t in traces}
        if len(actions) > 1:
            return i
    return min_len


# ---------------------------------------------------------------------------
# TracesPool
# ---------------------------------------------------------------------------

class TracesPool:
    """
    Collects episode traces and compresses them in a background thread.

    Trigger logic (checked after each episode):
      - capacity:         total compressed steps >= capacity_threshold
      - performance_drop: consecutive failures >= failure_streak_threshold
    """

    def __init__(
        self,
        capacity_threshold: int = 500,
        failure_streak_threshold: int = 5,
        loop_filter_window: int = 6,
        on_trigger=None,   # callable(reason, batch: dict) or None
    ):
        self.capacity_threshold = capacity_threshold
        self.failure_streak_threshold = failure_streak_threshold
        self.loop_filter_window = loop_filter_window
        self.on_trigger = on_trigger

        self._lock = threading.Lock()
        self._traces: list[Trace] = []
        self._compressed_step_count = 0
        self._consecutive_failures = 0

        self._compress_queue: list[Trace] = []
        self._compress_thread = threading.Thread(target=self._compress_worker, daemon=True)
        self._compress_event = threading.Event()
        self._compress_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, trace: Trace) -> None:
        """Add a completed episode trace; compression runs in background."""
        with self._lock:
            self._compress_queue.append(trace)
        self._compress_event.set()

    def get_batch(self, task_type: Optional[str] = None) -> dict:
        """
        Return a dict with success/failure trace lists, optionally filtered
        by task_type. Does NOT drain the pool.
        """
        with self._lock:
            traces = [t for t in self._traces if task_type is None or t.task_type == task_type]
        success = [t for t in traces if t.outcome == "success"]
        failure = [t for t in traces if t.outcome != "success"]
        return {"success": success, "failure": failure, "task_type": task_type}

    def drain(self) -> list[Trace]:
        """Return and clear all compressed traces."""
        with self._lock:
            out = list(self._traces)
            self._traces.clear()
            self._compressed_step_count = 0
            self._consecutive_failures = 0
            return out

    def stats(self) -> dict:
        with self._lock:
            by_type: dict[str, dict] = defaultdict(lambda: {"success": 0, "failure": 0})
            for t in self._traces:
                key = "success" if t.outcome == "success" else "failure"
                by_type[t.task_type][key] += 1
            return {
                "total_traces": len(self._traces),
                "compressed_steps": self._compressed_step_count,
                "consecutive_failures": self._consecutive_failures,
                "by_task_type": dict(by_type),
            }

    # ------------------------------------------------------------------
    # Background compression worker
    # ------------------------------------------------------------------

    def _compress_worker(self) -> None:
        while True:
            self._compress_event.wait()
            self._compress_event.clear()
            while True:
                with self._lock:
                    if not self._compress_queue:
                        break
                    trace = self._compress_queue.pop(0)

                compressed = self._compress(trace)

                with self._lock:
                    self._traces.append(compressed)
                    self._compressed_step_count += len(compressed.effective_steps)
                    if compressed.outcome != "success":
                        self._consecutive_failures += 1
                    else:
                        self._consecutive_failures = 0
                    reason = self._check_trigger()

                if reason and self.on_trigger:
                    batch = self.get_batch()
                    self.on_trigger(reason, batch)

    def _compress(self, trace: Trace) -> Trace:
        steps = trace.steps
        steps = _diff_compress(steps)
        steps = _loop_filter(steps, self.loop_filter_window)
        trace.compressed_steps = steps
        return trace

    def _check_trigger(self) -> Optional[str]:
        if self._compressed_step_count >= self.capacity_threshold:
            return "capacity"
        if self._consecutive_failures >= self.failure_streak_threshold:
            return "performance_drop"
        return None

    # ------------------------------------------------------------------
    # Prefix-tree analysis (called externally before cloud upload)
    # ------------------------------------------------------------------

    def get_shared_prefix_length(self, task_type: Optional[str] = None) -> int:
        """Return the common action prefix length across all traces of a task type."""
        with self._lock:
            traces = [t for t in self._traces
                      if task_type is None or t.task_type == task_type]
        return _shared_prefix_length(traces)

    def group_by_prefix(self, task_type: Optional[str] = None) -> dict[str, list[Trace]]:
        """
        Group traces by their first diverging action (after shared prefix).
        Useful for contrastive distillation: each group shares the same start.
        """
        with self._lock:
            traces = [t for t in self._traces
                      if task_type is None or t.task_type == task_type]
        if not traces:
            return {}
        prefix_len = _shared_prefix_length(traces)
        groups: dict[str, list[Trace]] = defaultdict(list)
        for t in traces:
            steps = t.effective_steps
            key = steps[prefix_len].action if prefix_len < len(steps) else "__end__"
            groups[key].append(t)
        return dict(groups)
