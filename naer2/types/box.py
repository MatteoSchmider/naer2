import torch


class _BaseBox:
    """
    An axis-aligned bounding box in a neuron coordinate space.

    Stores the extent of a region as a [ndim, 2] tensor where each row is
    [start, end] for one spatial dimension. This is the primary Region type
    for populations with regular grid structure:
      - A flat 1D population of N neurons: [[0, N]]
      - A 2D cortical sheet of shape (H, W): [[0, H], [0, W]]

    Boxes serve two roles:
      1. Population shape — the full extent of a population's neuron space.
      2. Pattern return value — a rectangular subregion of the target space
         that a given source neuron connects to (e.g. a local receptive field).

    When a pattern returns a Box, the JIT compiler can compute weight offsets
    without an explicit adjacency matrix, enabling procedural connectivity.
    If returned target coordinates fall outside the declared target population,
    new neurons can be auto-generated, which enables implicit layer construction
    (convolution, pooling) from the pattern alone.

    Tensor shape: [ndim, 2], dtype selects IndexBox vs ContinuousBox.
    """

    def __init__(self, coords: torch.Tensor):
        self._coords = None
        self.coords = coords  # validated through setter

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor):
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{type(self).__name__} coords must be shape [ndim, 2]")
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
    def _check_dtype(_: torch.Tensor):
        raise NotImplementedError


class IndexBox(_BaseBox):
    """Integer-coordinate box — default type for index-addressed populations."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
            raise TypeError("IndexBox requires integer tensor")


class ContinuousBox(_BaseBox):
    """Float-coordinate box — for spatially embedded populations with continuous coordinates."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise TypeError("ContinuousBox requires floating tensor")


class Box:
    """
    Factory returning IndexBox or ContinuousBox depending on tensor dtype.

    Integer tensors → IndexBox. Float tensors → ContinuousBox.
    Raises TypeError if the tensor dtype matches neither.
    """

    def __new__(cls, tensor: torch.Tensor):
        try:
            return IndexBox(tensor)
        except TypeError:
            return ContinuousBox(tensor)
