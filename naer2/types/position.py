import torch
from abc import ABC


class Position(ABC):
    """
    A single point in a neuron's coordinate space.

    Position is the source_id passed to a pattern callback — it identifies
    one specific neuron within a source Region. The concrete subtype mirrors
    the Region it is drawn from: neurons in an IndexBox are identified by
    IndexPosition, neurons in a ContinuousBox by ContinuousPosition.

    Tensor shape: rank-1, [ndim], where ndim is the dimensionality of the
    coordinate space (1 for a flat population, 2 for a cortical sheet, etc.).

    Position is an ABC so that isinstance(x, Position) works for type
    verification in pattern Hook.verify() and connection validation.
    """


class _BasePosition(Position):
    """
    Shared implementation for Position subtypes.

    Wraps a rank-1 torch.Tensor and delegates attribute access to it,
    so positions can be used directly in tensor arithmetic within pattern
    callbacks (e.g. computing offsets, distances).
    """

    def __init__(self, coords: torch.Tensor):
        self._coords: torch.Tensor = None
        self.coords = coords  # validated through setter

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor) -> None:
        if value.ndim != 1:
            raise ValueError(f"{type(self).__name__} must be rank-1 (vector)")
        self._check_dtype(value)
        self._coords = value

    def __getitem__(self, idx):
        return self._coords[idx]

    def __setitem__(self, idx, value):
        self._coords[idx] = value

    def __getattr__(self, name):
        return getattr(self._coords, name)

    def __repr__(self):
        return f"{type(self).__name__}({self._coords})"

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        raise NotImplementedError


class IndexPosition(_BasePosition):
    """Integer-coordinate position — used with IndexBox regions."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
            raise TypeError("IndexPosition requires integer tensor")


class ContinuousPosition(_BasePosition):
    """Float-coordinate position — used with ContinuousBox regions."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise TypeError("ContinuousPosition requires floating tensor")
