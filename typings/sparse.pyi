import numpy.typing
import numpy

class COO:
    @classmethod
    def from_numpy(
        cls,
        x: numpy.typing.NDArray[numpy.generic],
        fill_value: float | None = None,
        idx_dtype: numpy.generic | None = None,
    ) -> COO: ...
    ...
