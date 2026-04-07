import triton.language as tl


class _BaseBox:
    def __init__(self, coords: tl.tensor):
        self._coords = None
        self.coords = coords  # validated

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value: tl.tensor):
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{type(self).__name__} must be shape [ndim, 2]")
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


class IndexBox(_BaseBox):
    @staticmethod
    def _check_dtype(coords: tl.tensor):
        if coords.dtype not in [tl.int8, tl.int16, tl.int32, tl.int64]:
            raise TypeError("IndexBox requires integer tensor")


class ContinuousBox(_BaseBox):
    @staticmethod
    def _check_dtype(coords: tl.tensor):
        if coords.dtype not in [tl.float16, tl.float32, tl.float64]:
            raise TypeError("ContinuousBox requires floating tensor")


Box = IndexBox | ContinuousBox
