import torch


class _BasePoints:
    """
    An explicit enumeration of neuron positions in coordinate space.

    Stores an arbitrary set of points as a [n_points, ndim] tensor — used
    when a population's neurons have irregular or biologically-derived positions
    that cannot be described by a bounding box.

    Iteration and position_at(i) yield each stored point as a Position.
    position_at(i) is a simple gather (coords[i]) — directly GPU-parallelizable
    by substituting a Triton program_id for i and loading from the coords buffer.

    Tensor shape: [n_points, ndim].
    """

    def __init__(self, coords: torch.Tensor):
        self._coords = None
        self.coords = coords  # validated through setter

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor):
        if value.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} coords must be rank-2 [n_points, ndim], "
                f"got ndim={value.ndim}"
            )
        self._check_dtype(value)
        self._coords = value

    # ---------------------------------------------------------------------- iteration

    def __len__(self) -> int:
        return int(self._coords.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self.position_at(i)

    def position_at(self, i: int):
        """
        Return the Position at row i.

        Pure gather — GPU-parallelizable by substituting a Triton program_id
        for i and loading coords[i] from the preloaded coords buffer.
        """
        return self._make_position(self._coords[i])

    # ---------------------------------------------------------------------- index / proxy

    def __getitem__(self, idx):
        return self._coords[idx]

    def __setitem__(self, idx, value):
        self._coords[idx] = value

    def __getattr__(self, name):
        return getattr(self._coords, name)

    def __repr__(self):
        return f"{type(self).__name__}({self._coords})"

    # ---------------------------------------------------------------------- abstract

    @staticmethod
    def _check_dtype(_: torch.Tensor):
        raise NotImplementedError

    def _make_position(self, _: torch.Tensor):
        raise NotImplementedError


class IndexPoints(_BasePoints):
    """Integer-coordinate point set — discrete neuron indices."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("IndexPoints requires integer tensor")

    def _make_position(self, coords: torch.Tensor):
        from .position import IndexPosition
        return IndexPosition(coords)


class ContinuousPoints(_BasePoints):
    """Float-coordinate point set — spatially embedded neuron positions."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("ContinuousPoints requires floating tensor")

    def _make_position(self, coords: torch.Tensor):
        from .position import ContinuousPosition
        return ContinuousPosition(coords)


class Points:
    """
    Factory returning IndexPoints or ContinuousPoints depending on tensor dtype.

    Integer tensors → IndexPoints. Float tensors → ContinuousPoints.
    """

    def __new__(cls, tensor: torch.Tensor):
        try:
            return IndexPoints(tensor)
        except TypeError:
            return ContinuousPoints(tensor)
