import triton.language as tl


class _BasePosition:
    def __init__(self, coords: tl.tensor):
        self._coords: tl.tensor = None
        self.coords = coords  # validated through setter

    @property
    def coords(self) -> tl.tensor:
        return self._coords

    @coords.setter
    def coords(self, value: tl.tensor) -> None:
        if value.ndim != 1:
            raise ValueError(f"{type(self).__name__} must be rank-1 (vector)")
        self._check_dtype(value)
        self._coords = value

    def __getitem__(self, idx):
        return self._coords[idx]

    def __setitem__(self, idx, value):
        self._coords[idx] = value

    def __getattr__(self, name):
        # Forward unknown attributes to tensor
        return getattr(self._coords, name)

    def __repr__(self):
        return f"{type(self).__name__}({self._coords})"

    @staticmethod
    def _check_dtype(coords: tl.tensor):
        raise NotImplementedError


class IndexPosition(_BasePosition):
    @staticmethod
    def _check_dtype(coords: tl.tensor):
        if coords.dtype not in [tl.int8, tl.int16, tl.int32, tl.int64]:
            raise TypeError("IndexPosition requires integer tensor")


class ContinuousPosition(_BasePosition):
    @staticmethod
    def _check_dtype(coords: tl.tensor):
        if coords.dtype not in [tl.float16, tl.float32, tl.float64]:
            raise TypeError("ContinuousPosition requires floating tensor")


Position = IndexPosition | ContinuousPosition
