from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Iterator, TypeVar

try:
    from tqdm.auto import tqdm as _auto_tqdm
except Exception:  # pragma: no cover - tqdm fallback depends on environment
    _auto_tqdm = None

T = TypeVar("T")


class _PlainProgressBar:
    def __init__(
        self,
        total: int | None,
        description: str,
        unit: str = "step",
        emit_interval_seconds: float = 5.0,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.total = total
        self.description = description
        self.unit = unit
        self.n = 0
        self._clock = clock
        self._emit_interval_seconds = max(float(emit_interval_seconds), 0.0)
        self._started_at = self._clock()
        self._last_emitted_at = self._started_at
        self._last_emitted = -1
        self._emit("START")

    def _should_emit(self) -> bool:
        if self.total in (None, 0):
            return True
        if self.total <= 10:
            return True
        if self.n >= self.total:
            return True
        checkpoint = max(1, self.total // 10)
        checkpoint_reached = self.n // checkpoint > self._last_emitted // checkpoint
        heartbeat_due = self._clock() - self._last_emitted_at >= self._emit_interval_seconds
        return checkpoint_reached or heartbeat_due

    def _emit(self, state: str) -> None:
        elapsed = self._clock() - self._started_at
        if self.total:
            print(
                f"[{state}] {self.description} [{self.n}/{self.total} {self.unit}] "
                f"elapsed={elapsed:.1f}s"
            )
        else:
            print(f"[{state}] {self.description} [{self.n} {self.unit}] elapsed={elapsed:.1f}s")
        self._last_emitted = self.n
        self._last_emitted_at = self._clock()

    def update(self, increment: int = 1) -> None:
        self.n += increment
        if self._should_emit():
            self._emit("RUNNING")

    def set_description_str(self, description: str, refresh: bool = True) -> None:
        self.description = description
        if refresh:
            self._emit("RUNNING")

    def close(self) -> None:
        return None


@dataclass
class NotebookCellProgress:
    notebook_name: str
    cell_name: str
    total_steps: int
    progress_bar: object

    @property
    def base_description(self) -> str:
        return f"{self.notebook_name} | {self.cell_name}"


def _build_progress_bar(
    total: int | None,
    description: str,
    unit: str = "step",
    leave: bool = True,
):
    if _auto_tqdm is not None:
        return _auto_tqdm(total=total, desc=description, unit=unit, leave=leave, dynamic_ncols=True)
    return _PlainProgressBar(total=total, description=description, unit=unit)


def start_notebook_cell_progress(
    notebook_name: str,
    cell_name: str,
    total_steps: int = 1,
) -> NotebookCellProgress:
    handle = NotebookCellProgress(
        notebook_name=notebook_name,
        cell_name=cell_name,
        total_steps=total_steps,
        progress_bar=_build_progress_bar(
            total=total_steps,
            description=f"{notebook_name} | {cell_name}",
            unit="step",
            leave=True,
        ),
    )
    return handle


def advance_notebook_cell_progress(
    handle: NotebookCellProgress,
    step_label: str | None = None,
    increment: int = 1,
) -> NotebookCellProgress:
    if step_label:
        handle.progress_bar.set_description_str(f"{handle.base_description} | {step_label}")
    handle.progress_bar.update(increment)
    return handle


def finish_notebook_cell_progress(
    handle: NotebookCellProgress,
    final_label: str = "done",
) -> None:
    if final_label:
        handle.progress_bar.set_description_str(f"{handle.base_description} | {final_label}")

    remaining = max(int(handle.total_steps) - int(getattr(handle.progress_bar, "n", 0)), 0)
    if remaining:
        handle.progress_bar.update(remaining)
    handle.progress_bar.close()


def iter_notebook_progress(
    iterable: Iterable[T],
    description: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = False,
) -> Iterator[T]:
    if _auto_tqdm is not None:
        yield from _auto_tqdm(iterable, total=total, desc=description, unit=unit, leave=leave, dynamic_ncols=True)
        return

    progress_bar = _PlainProgressBar(total=total, description=description, unit=unit)
    for item in iterable:
        yield item
        progress_bar.update(1)
    progress_bar.close()


__all__ = [
    "NotebookCellProgress",
    "advance_notebook_cell_progress",
    "finish_notebook_cell_progress",
    "iter_notebook_progress",
    "start_notebook_cell_progress",
]
