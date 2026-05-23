from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from config import SIDO_NAME_STANDARDIZATION
from src.qa import check_required_columns
from src.utils import (
    build_key_series,
    cast_float_columns,
    cast_nullable_int_columns,
    cast_string_columns,
    clean_text,
    remove_newlines_from_object_columns,
    reorder_columns,
    standardize_name_values,
    trim_object_columns,
)

GUSIGUN_REGION_FALLBACK = {
    "세종특별자치시": "충청권",
    "제주특별자치도": "제주권",
}

DIM_DONG_COLUMNS = [
    "읍면동KEY",
    "구시군KEY",
    "기준시점",
    "행정기관코드",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "국회의원선거구(2024)",
    "광역의원선거구(2022)",
    "기초의원선거구(2022)",
    "비고",
]

DIM_GUSIGUN_COLUMNS = [
    "구시군KEY",
    "기준시점",
    "행정기관코드",
    "권역",
    "시도",
    "구시군",
]

DIM_ELECTION_COLUMNS = [
    "선거KEY",
    "선거시점",
    "선거타입",
    "비례여부",
    "선거종류",
    "선거명",
    "단위",
    "최소단위",
]

DIM_PARTY_COLUMNS = [
    "선거KEY",
    "정당KEY",
    "선거시점",
    "연번",
    "정당명",
    "구분",
    "구분2",
    "성향",
    "IsIndependent",
]

DIM_POLLING_PLACE_SOURCE_COLUMNS = [
    "선거시점",
    "선거타입",
    "시도명_F",
    "구시군명_F",
    "일반구명",
    "투표소명",
    "장소명",
    "주소",
    "읍면동명_F",
    "층수",
    "법정동",
    "위도",
    "경도",
    "시도명_D",
    "구시군명_D",
    "일반구명_D",
    "읍면동명_D",
    "읍면동KEY_D",
    "투표소KEY",
]

DIM_POLLING_PLACE_COLUMNS = [
    "선거시점",
    "선거타입",
    "시도명_F",
    "구시군명_F",
    "일반구명_F",
    "투표소명_F",
    "장소명",
    "주소",
    "읍면동명_F",
    "층수",
    "법정동",
    "위도",
    "경도",
    "시도명_D",
    "구시군명_D",
    "일반구명_D",
    "읍면동명_D",
    "읍면동KEY_D",
    "투표소KEY",
]

DIM_DONG_ALIAS_COLUMNS = [
    "시도명_F",
    "구시군명_F",
    "일반구명_F",
    "읍면동명_F",
    "시도명_D",
    "구시군명_D",
    "일반구명_D",
    "읍면동명_D",
    "읍면동KEY_F",
    "읍면동KEY_D",
]


def transform_dim_dong(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, DIM_DONG_COLUMNS, "DimDong")

    result = trim_object_columns(df)
    result = standardize_name_values(result, ["시도명"], SIDO_NAME_STANDARDIZATION)
    return reorder_columns(result, DIM_DONG_COLUMNS, "DimDong")


def transform_dim_gusigun(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, DIM_GUSIGUN_COLUMNS, "DimGusigun")

    result = trim_object_columns(df)
    result = standardize_name_values(result, ["시도"], SIDO_NAME_STANDARDIZATION)
    result = cast_string_columns(result, ["구시군KEY", "권역", "시도", "구시군"])
    result = cast_nullable_int_columns(result, ["기준시점", "행정기관코드"])
    return reorder_columns(result, DIM_GUSIGUN_COLUMNS, "DimGusigun")


def transform_dim_election(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, DIM_ELECTION_COLUMNS, "DimElection")

    result = trim_object_columns(df)
    result = cast_string_columns(result, ["선거KEY", "선거종류", "선거명", "단위", "최소단위"])
    result = cast_nullable_int_columns(result, ["선거시점", "비례여부"])
    if "선거타입" in result.columns:
        result["선거타입"] = result["선거타입"].map(clean_text)
    return reorder_columns(result, DIM_ELECTION_COLUMNS, "DimElection")


def transform_dim_party(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, DIM_PARTY_COLUMNS, "DimParty")

    result = trim_object_columns(df)
    result = cast_string_columns(result, ["선거KEY", "정당KEY", "정당명", "구분", "구분2", "성향"])
    result = cast_nullable_int_columns(result, ["선거시점", "연번", "IsIndependent"])
    return reorder_columns(result, DIM_PARTY_COLUMNS, "DimParty")


def _prepare_dim_polling_place(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) == DIM_POLLING_PLACE_COLUMNS:
        return df.reset_index(drop=True)

    if list(df.columns) == DIM_POLLING_PLACE_SOURCE_COLUMNS:
        result = df.copy()
        result.columns = DIM_POLLING_PLACE_COLUMNS
        return result.reset_index(drop=True)

    if len(df.columns) == len(DIM_POLLING_PLACE_COLUMNS) and all(
        isinstance(column, int) for column in df.columns
    ):
        result = df.iloc[1:].copy()
        result.columns = DIM_POLLING_PLACE_COLUMNS
        return result.reset_index(drop=True)

    if not df.empty:
        first_row = [clean_text(value) for value in df.iloc[0].tolist()]
        if first_row == DIM_POLLING_PLACE_SOURCE_COLUMNS:
            result = df.iloc[1:].copy()
            result.columns = DIM_POLLING_PLACE_COLUMNS
            return result.reset_index(drop=True)

    raise ValueError("DimPollingPlace: unexpected source schema.")


def transform_dim_polling_place(df: pd.DataFrame) -> pd.DataFrame:
    result = _prepare_dim_polling_place(df)
    check_required_columns(result, DIM_POLLING_PLACE_COLUMNS, "DimPollingPlace")

    result = trim_object_columns(result)
    result = standardize_name_values(result, ["시도명_F", "시도명_D"], SIDO_NAME_STANDARDIZATION)
    result = cast_float_columns(result, ["위도", "경도"])
    return reorder_columns(result, DIM_POLLING_PLACE_COLUMNS, "DimPollingPlace")


def transform_dim_dong_alias(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, DIM_DONG_ALIAS_COLUMNS, "DimDongAlias")

    result = remove_newlines_from_object_columns(df)
    result = standardize_name_values(result, ["시도명_F", "시도명_D"], SIDO_NAME_STANDARDIZATION)
    return reorder_columns(result, DIM_DONG_ALIAS_COLUMNS, "DimDongAlias")


def _supplement_dim_dong_from_alias(dim_dong: pd.DataFrame, dim_dong_alias: pd.DataFrame) -> pd.DataFrame:
    if dim_dong.empty or dim_dong_alias.empty:
        return dim_dong

    alias_rows = dim_dong_alias.loc[
        :,
        ["시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D", "읍면동KEY_D"],
    ].drop_duplicates(ignore_index=True)
    existing_keys = set(dim_dong["읍면동KEY"].dropna().astype("string"))
    alias_rows = alias_rows.loc[~alias_rows["읍면동KEY_D"].astype("string").isin(existing_keys)].copy()
    if alias_rows.empty:
        return dim_dong

    existing_labels = dim_dong.loc[:, ["시도명", "구시군명", "일반구명", "읍면동명"]].drop_duplicates(ignore_index=True)
    alias_rows = alias_rows.merge(
        existing_labels,
        left_on=["시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D"],
        right_on=["시도명", "구시군명", "일반구명", "읍면동명"],
        how="left",
        indicator=True,
    )
    alias_rows = alias_rows.loc[alias_rows["_merge"] != "both", ["시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D", "읍면동KEY_D"]]
    if alias_rows.empty:
        return dim_dong

    supplemental = pd.DataFrame(
        {
            "읍면동KEY": alias_rows["읍면동KEY_D"].astype("string"),
            "구시군KEY": build_key_series(alias_rows["시도명_D"], alias_rows["구시군명_D"]),
            "기준시점": pd.Series(pd.NA, index=alias_rows.index, dtype="Int32"),
            "행정기관코드": pd.Series(pd.NA, index=alias_rows.index, dtype="Int32"),
            "시도명": alias_rows["시도명_D"].astype("string"),
            "구시군명": alias_rows["구시군명_D"].astype("string"),
            "일반구명": alias_rows["일반구명_D"].astype("string"),
            "읍면동명": alias_rows["읍면동명_D"].astype("string"),
            "국회의원선거구(2024)": pd.Series(pd.NA, index=alias_rows.index, dtype="string"),
            "광역의원선거구(2022)": pd.Series(pd.NA, index=alias_rows.index, dtype="string"),
            "기초의원선거구(2022)": pd.Series(pd.NA, index=alias_rows.index, dtype="string"),
            "비고": pd.Series("DimDongAlias 보강", index=alias_rows.index, dtype="string"),
        }
    )
    combined = pd.concat([dim_dong, supplemental], ignore_index=True)
    return reorder_columns(combined.drop_duplicates(subset=["읍면동KEY"], keep="first"), DIM_DONG_COLUMNS, "DimDong")


def _build_region_map(dim_gusigun: pd.DataFrame) -> dict[str, str]:
    if dim_gusigun.empty:
        return dict(GUSIGUN_REGION_FALLBACK)

    region_map = (
        dim_gusigun.loc[dim_gusigun["권역"].astype("string").notna(), ["시도", "권역"]]
        .drop_duplicates()
        .groupby("시도", dropna=False)["권역"]
        .agg(lambda values: values.iloc[0])
        .to_dict()
    )
    for sido, region in GUSIGUN_REGION_FALLBACK.items():
        region_map.setdefault(sido, region)
    return region_map


def _derive_gusigun_admin_code(series: pd.Series) -> object:
    values = series.dropna().astype("Int64")
    if values.empty:
        return pd.NA

    code_text = str(int(values.iloc[0]))
    if len(code_text) < 5:
        return pd.NA
    return int(f"{code_text[:5]}00000")


def _supplement_dim_gusigun_from_dong(dim_gusigun: pd.DataFrame, dim_dong: pd.DataFrame) -> pd.DataFrame:
    if dim_dong.empty:
        return dim_gusigun

    existing_keys = set(dim_gusigun["구시군KEY"].dropna().astype("string"))
    grouped = (
        dim_dong.loc[:, ["구시군KEY", "기준시점", "행정기관코드", "시도명", "구시군명"]]
        .dropna(subset=["구시군KEY"])
        .groupby("구시군KEY", dropna=False, as_index=False)
        .agg(
            기준시점=("기준시점", "max"),
            시도=("시도명", "first"),
            구시군=("구시군명", "first"),
            행정기관코드=("행정기관코드", _derive_gusigun_admin_code),
        )
    )
    grouped = grouped.loc[~grouped["구시군KEY"].astype("string").isin(existing_keys)].copy()
    if grouped.empty:
        return dim_gusigun

    region_map = _build_region_map(dim_gusigun)
    grouped["권역"] = grouped["시도"].map(region_map).astype("string")
    supplemental = grouped.loc[:, ["구시군KEY", "기준시점", "행정기관코드", "권역", "시도", "구시군"]]
    combined = pd.concat([dim_gusigun, supplemental], ignore_index=True)
    return reorder_columns(combined.drop_duplicates(subset=["구시군KEY"], keep="first"), DIM_GUSIGUN_COLUMNS, "DimGusigun")


def transform_all_dims(dim_tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    transformed: dict[str, pd.DataFrame] = {}

    if "DimDong" in dim_tables:
        transformed["DimDong"] = transform_dim_dong(dim_tables["DimDong"])
    if "DimElection" in dim_tables:
        transformed["DimElection"] = transform_dim_election(dim_tables["DimElection"])
    if "DimParty" in dim_tables:
        transformed["DimParty"] = transform_dim_party(dim_tables["DimParty"])
    if "DimPollingPlace" in dim_tables:
        transformed["DimPollingPlace"] = transform_dim_polling_place(dim_tables["DimPollingPlace"])
    if "DimDongAlias" in dim_tables:
        transformed["DimDongAlias"] = transform_dim_dong_alias(dim_tables["DimDongAlias"])
    if "DimDong" in transformed and "DimDongAlias" in transformed:
        transformed["DimDong"] = _supplement_dim_dong_from_alias(transformed["DimDong"], transformed["DimDongAlias"])
    if "DimGusigun" in dim_tables:
        transformed["DimGusigun"] = transform_dim_gusigun(dim_tables["DimGusigun"])
    if "DimGusigun" in transformed and "DimDong" in transformed:
        transformed["DimGusigun"] = _supplement_dim_gusigun_from_dong(transformed["DimGusigun"], transformed["DimDong"])

    return transformed
