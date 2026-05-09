import torch


class _BasePoints:
    """
    An explicit enumeration of neuron positions in coordinate space.

    Stores an arbitrary set of points as a [n_points, ndim] tensor — used
    when a population's neurons have irregular or biologically-derived positions
    that cannot be described by a bounding box. For example, neurons placed at
    measured 3D coordinates from a biological dataset.

    As a pattern return value, Points allows a source neuron to connect to a
    non-rectangular, explicitly listed set of target neurons. Unlike Box,
    this cannot generally be used for procedural weight addressing — the JIT
    compiler will fall back to generating an explicit adjacency structure.

    Tensor shape: [n_points, ndim].
    """

    def __init__(self, coords: torch.Tensor):
        self._coords = None
        self.coords = coords  # validated through setter

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor):
        if value.ndim != 2:
            raise ValueError(f"{type(self).__name__} coords must be rank-2 [n_points, ndim]")
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


class IndexPoints(_BasePoints):
    """Integer-coordinate point set — discrete neuron indices."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
            raise TypeError("IndexPoints requires integer tensor")


class ContinuousPoints(_BasePoints):
    """Float-coordinate point set — spatially embedded neuron positions."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise TypeError("ContinuousPoints requires floating tensor")


class Points:
    """
    Factory returning IndexPoints or ContinuousPoints depending on tensor dtype.

    Integer tensors → IndexPoints. Float tensors → ContinuousPoints.
    Raises TypeError if the tensor dtype matches neither.
    """

    def __new__(cls, tensor: torch.Tensor):
        try:
            return IndexPoints(tensor)
        except TypeError:
            return ContinuousPoints(tensor)
