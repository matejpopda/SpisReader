"""
Module contains tools for processing files into DataFrames or other objects

GH#48849 provides a convenient way of deprecating keyword arguments
"""

from __future__ import annotations

import csv
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
)


from pandas._libs import lib


from pandas.core.frame import DataFrame


if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        ReadCsvBuffer,
        StorageOptions,
    )


def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: Any = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable[..., Any]] | None = ...,
    true_values: list[Hashable] | None = ...,
    false_values: list[Hashable] | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: (
        Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None
    ) = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable[..., Any] | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines: Literal["error", "warn", "skip"] = ...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...
