from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd

from config import SIDO_NAME_STANDARDIZATION

SPECIAL_WHITESPACE_PATTERN = re.compile(r"[\u00a0\u2007\u202f\ufeff\t]+")
MULTI_SPACE_PATTERN = re.compile(r" {2,}")


def strip_special_whitespace(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if not isinstance(value, str):
        return value

    cleaned = SPECIAL_WHITESPACE_PATTERN.sub(" ", value)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.strip()
    return cleaned or pd.NA


def remove_linebreaks(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if not isinstance(value, str):
        return value

    cleaned = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return strip_special_whitespace(cleaned)


def clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if not isinstance(value, str):
        return value
    return remove_linebreaks(value)


def clean_series(series: pd.Series, remove_breaks: bool = True) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    cleaned = series.astype("string")
    if remove_breaks:
        cleaned = cleaned.str.replace("\r\n", " ", regex=False)
        cleaned = cleaned.str.replace("\r", " ", regex=False)
        cleaned = cleaned.str.replace("\n", " ", regex=False)
    cleaned = cleaned.str.replace(SPECIAL_WHITESPACE_PATTERN, " ", regex=True)
    cleaned = cleaned.str.replace(MULTI_SPACE_PATTERN, " ", regex=True)
    cleaned = cleaned.str.strip()
    return cleaned.mask(cleaned.eq(""), pd.NA)


def clean_text_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    remove_breaks: bool = True,
) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = clean_series(df[column], remove_breaks=remove_breaks)
    return df


def normalize_sido_name(value: object) -> object:
    cleaned = clean_text(value)
    if isinstance(cleaned, str):
        return SIDO_NAME_STANDARDIZATION.get(cleaned, cleaned)
    return cleaned


def normalize_sido_series(series: pd.Series) -> pd.Series:
    cleaned = clean_series(series)
    if not (pd.api.types.is_object_dtype(cleaned) or pd.api.types.is_string_dtype(cleaned)):
        return cleaned
    return cleaned.replace(SIDO_NAME_STANDARDIZATION)


def compact_location_series(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    cleaned = series.astype("string")
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace("\t", "", regex=False)
    cleaned = cleaned.str.replace("\r", "", regex=False)
    cleaned = cleaned.str.replace("\n", "", regex=False)
    return cleaned.mask(cleaned.eq(""), pd.NA)


def build_key(parts: Iterable[object]) -> object:
    cleaned_parts: list[str] = []
    for part in parts:
        cleaned = clean_text(part)
        if isinstance(cleaned, str) and cleaned:
            cleaned_parts.append(cleaned)

    if not cleaned_parts:
        return pd.NA
    return " ".join(cleaned_parts)


def build_key_series(*parts: pd.Series) -> pd.Series:
    if not parts:
        raise ValueError("build_key_series requires at least one input series.")

    key = pd.Series("", index=parts[0].index, dtype="string")
    for part in parts:
        values = clean_series(part).astype("string")
        values = values.fillna("")
        key = key.where(values.eq("") | key.eq(""), key + " ")
        key = key + values

    key = key.str.strip()
    return key.mask(key.eq(""), pd.NA)


def safe_int_cast(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64")

    cleaned = clean_series(series, remove_breaks=False).astype("string")
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


def safe_float_cast(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")

    cleaned = clean_series(series, remove_breaks=False).astype("string")
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def trim_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text_columns = result.select_dtypes(include=["object", "string"]).columns
    return clean_text_columns(result, text_columns, remove_breaks=False)


def remove_newlines_from_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text_columns = result.select_dtypes(include=["object", "string"]).columns
    return clean_text_columns(result, text_columns, remove_breaks=True)


def cast_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = clean_series(result[column]).astype("string")
    return result


def cast_nullable_int_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = safe_int_cast(result[column])
    return result


def cast_float_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = safe_float_cast(result[column])
    return result


def reorder_columns(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{name}: missing columns for reordering: {missing_columns}")
    return df.loc[:, columns].copy()


def standardize_name_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    mapping: Mapping[str, str],
) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].replace(mapping)
    return result
