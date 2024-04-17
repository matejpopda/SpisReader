from __future__ import annotations

import os
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)


from xarray.backends.common import (
    AbstractDataStore,
)
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    from io import BufferedIOBase

    from xarray.backends.common import BackendEntrypoint
    from xarray.core.types import (
        T_Chunks,
    )

    T_NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
    T_Engine = Union[
        T_NetcdfEngine,
        Literal["pydap", "pynio", "zarr"],
        type[BackendEntrypoint],
        str,  # no nice typing support for custom backends
        None,
    ]
    T_NetcdfTypes = Literal[
        "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
    ]


def open_dataset(
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    engine: T_Engine = None,
    chunks: T_Chunks = None,
    cache: bool | None = None,
    decode_cf: bool | None = None,
    mask_and_scale: bool | None = None,
    decode_times: bool | None = None,
    decode_timedelta: bool | None = None,
    use_cftime: bool | None = None,
    concat_characters: bool | None = None,
    decode_coords: Literal["coordinates", "all"] | bool | None = None,
    drop_variables: str | Iterable[str] | None = None,
    inline_array: bool = False,
    chunked_array_type: str | None = None,
    from_array_kwargs: dict[str, Any] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dataset: ...
