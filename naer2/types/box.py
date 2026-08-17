import torch


class _BaseBox:
    """
    An axis-aligned bounding box in a neuron coordinate space.

    Stores the extent of a region as a [ndim, 3] tensor where each row is
    [start, stop, step] for one spatial dimension. If constructed from a
    [ndim, 2] tensor, a default step column is appended automatically.

    Iteration and position_at(i) expose every lattice point in the box in
    row-major order (last dimension varies fastest). For a 1D IndexBox [0, N, 1]
    this is just the N integers 0 … N-1. For nD it is the Cartesian product.

    position_at(i) is designed to be GPU-parallel: replace i with a Triton
    program_id and the computation is pure arithmetic — no branches, no loops.
    For IndexBox that means integer modulo + multiply-add; for ContinuousBox
    the same in floating point.

    Boxes serve two roles:
      1. Population shape — the full extent of a population's neuron space.
      2. Pattern return value — a rectangular subregion of the target space.
    """

    def __init__(self, coords: torch.Tensor):
        self._coords = None
        self.coords = coords  # validated through setter

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @coords.setter
    def coords(self, value: torch.Tensor):
        if value.ndim != 2 or value.shape[1] not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} coords must be shape [ndim, 2] or [ndim, 3], "
                f"got {tuple(value.shape)}"
            )
        self._check_dtype(value)
        if value.shape[1] == 2:
            step = self._default_step(value)
            self._coords = torch.cat([value, step.unsqueeze(1)], dim=1)
        else:
            self._coords = value

    @property
    def sizes(self) -> torch.Tensor:
        """Number of lattice points per dimension."""
        return torch.ceil(
            (self._coords[:, 1].float() - self._coords[:, 0].float())
            / self._coords[:, 2].float()
        ).to(torch.int64)

    # ---------------------------------------------------------------------- iteration

    def __len__(self) -> int:
        return int(self.sizes.prod().item())

    @property
    def points(self):
        return torch.cartesian_prod(
            *[
                torch.arange(self.coords[d, 0], self.coords[d, 1], self.coords[d, 2])
                for d in range(self.coords.shape[0])
            ]
        )

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

    @staticmethod
    def _default_step(_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _make_position(self, coords: torch.Tensor):
        raise NotImplementedError


class IndexBox(_BaseBox):
    """Integer-coordinate box — default type for index-addressed populations."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("IndexBox requires integer tensor")

    @staticmethod
    def _default_step(coords: torch.Tensor) -> torch.Tensor:
        return torch.ones(coords.shape[0], dtype=torch.int64)

    def _make_position(self, coords: torch.Tensor):
        from .position import IndexPosition

        return IndexPosition(coords)


class ContinuousBox(_BaseBox):
    """Float-coordinate box — for spatially embedded populations with continuous coordinates."""

    @staticmethod
    def _check_dtype(coords: torch.Tensor):
        if coords.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("ContinuousBox requires floating tensor")

    @staticmethod
    def _default_step(coords: torch.Tensor) -> torch.Tensor:
        return torch.ones(coords.shape[0], dtype=coords.dtype)

    def _make_position(self, coords: torch.Tensor):
        from .position import ContinuousPosition

        return ContinuousPosition(coords)


class Box:
    """
    Factory returning IndexBox or ContinuousBox depending on tensor dtype.

    Accepts [ndim, 2] (start, stop) or [ndim, 3] (start, stop, step) tensors.
    Integer tensors → IndexBox. Float tensors → ContinuousBox.
    """

    def __new__(cls, tensor: torch.Tensor):
        try:
            return IndexBox(tensor)
        except TypeError:
            return ContinuousBox(tensor)
