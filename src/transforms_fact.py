from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import pandas as pd

from src.loaders import (
    FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS,
    FACT_RESIDENT_COMPOSITION_RAW_COLUMNS,
    FACT_TURNOUT_COLUMNS,
    FACT_VOTES_RAW_COLUMNS,
)
from src.qa import check_required_columns
from src.utils import (
    build_key_series,
    clean_series,
    clean_text,
    clean_text_columns,
    compact_location_series,
    normalize_sido_series,
    reorder_columns,
    safe_float_cast,
    safe_int_cast,
)

FACT_TURNOUT_NUMERIC_COLUMNS = ["선거인수", "투표수", "유효투표수", "무효투표수", "기권수"]
FACT_VOTES_NUMERIC_COLUMNS = ["후보슬롯", "유효투표수", "득표수", "득표율"]
FACT_CONFIRMED_ELECTORATE_NUMERIC_COLUMNS = [
    "인구수",
    "인구수_재외국민",
    "인구수_외국인",
    "확정선거인수",
    "확정선거인수_재외국민",
    "확정선거인수_외국인",
    "확정선거인수_남",
    "확정선거인수_남_재외국민",
    "확정선거인수_남_외국인",
    "확정선거인수_여",
    "확정선거인수_여_재외국민",
    "확정선거인수_여_외국인",
    "거소투표신고인명부등재자수",
    "거소투표신고인명부등재자수_재외국민",
    "거소투표신고인명부등재자수_남",
    "거소투표신고인명부등재자수_남_재외국민",
    "거소투표신고인명부등재자수_여",
    "거소투표신고인명부등재자수_여_재외국민",
]
FACT_RESIDENT_EXACT_AGE_COLUMNS = [f"{age}세" for age in range(0, 100)] + ["100세이상"]
FACT_RESIDENT_COMPOSITION_NUMERIC_COLUMNS = [
    "선거연령기준",
    "총인구수",
    "세대수",
    "세대당인구",
    "남자인구수",
    "여자인구수",
    "남여비율",
    "평균연령",
    "남자평균연령",
    "여자평균연령",
    "전월인구수",
    "당월인구수",
    "인구증감",
    "인구증감률",
    "전월남자인구수",
    "전월여자인구수",
    "당월남자인구수",
    "당월여자인구수",
    "남자인구증감",
    "여자인구증감",
    "아동인구",
    "청소년인구",
    "청년인구",
    "고령인구",
    "1인가구수",
    "청년1인가구수_추정",
    "노년1인가구수_추정",
    "10대인구",
    "20대인구",
    "30대인구",
    "40대인구",
    "50대인구",
    "60대인구",
    "70대인구",
    "80대인구",
    "90대인구",
    "100세이상인구",
    "70세이상인구",
    *FACT_RESIDENT_EXACT_AGE_COLUMNS,
    "선거연령청년인구_추정",
    "남성구성비",
    "여성구성비",
    "선거연령청년구성비_추정",
    "30대구성비",
    "40대구성비",
    "50대구성비",
    "60대구성비",
    "70세이상구성비",
    "아동구성비",
    "청소년구성비",
    "청년구성비",
    "고령구성비",
    "1인가구비율",
    "청년1인가구비율_추정",
    "노년1인가구비율_추정",
    "1인가구연령통계가용여부",
]

FACT_TURNOUT_FINAL_COLUMNS = [
    "선거KEY",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "투표구명",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
    "투표소KEY",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
]

FACT_VOTES_FINAL_COLUMNS = [
    "선거KEY",
    "정당KEY",
    "선거구명",
    "시도명",
    "구시군명",
    "구시군명_원본",
    "일반구명",
    "읍면동명",
    "구분",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
    "투표소KEY",
    "후보슬롯",
    "정당명",
    "후보명",
    "후보라벨",
    "유효투표수",
    "득표수",
    "득표율",
]

FACT_CONFIRMED_ELECTORATE_FINAL_COLUMNS = [
    "선거KEY",
    "API선거ID",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "투표구명",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
    "투표소KEY",
    "인구수",
    "인구수_재외국민",
    "인구수_외국인",
    "확정선거인수",
    "확정선거인수_재외국민",
    "확정선거인수_외국인",
    "확정선거인수_남",
    "확정선거인수_남_재외국민",
    "확정선거인수_남_외국인",
    "확정선거인수_여",
    "확정선거인수_여_재외국민",
    "확정선거인수_여_외국인",
    "거소투표신고인명부등재자수",
    "거소투표신고인명부등재자수_재외국민",
    "거소투표신고인명부등재자수_남",
    "거소투표신고인명부등재자수_남_재외국민",
    "거소투표신고인명부등재자수_여",
    "거소투표신고인명부등재자수_여_재외국민",
]
FACT_RESIDENT_COMPOSITION_FINAL_COLUMNS = FACT_RESIDENT_COMPOSITION_RAW_COLUMNS.copy()

FACT_VOTES_TEXT_COLUMNS = [
    "선거KEY",
    "정당KEY",
    "선거구명",
    "시도명",
    "구시군명",
    "읍면동명",
    "구분",
    "정당명",
    "후보명",
    "후보라벨",
]

FACT_TURNOUT_TEXT_COLUMNS = ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명"]
FACT_CONFIRMED_ELECTORATE_TEXT_COLUMNS = ["선거KEY", "API선거ID", "시도명_API", "구시군명_API", "읍면동명_API", "투표구명_API"]
FACT_RESIDENT_COMPOSITION_TEXT_COLUMNS = [
    "선거KEY",
    "API선거ID",
    "선거일",
    "기준월",
    "기준월라벨",
    "행정기관코드",
    "행정구역명_API",
    "시도명_API",
    "구시군명_API",
    "일반구명_API",
    "읍면동명_API",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
]

TURNOUT_KEY_FIXUPS = {
    "201806-L3-": "201806-L3",
    "210806-L6": "201806-L6",
}

ABSENTEE_LABELS = {
    "관외사전투표",
    "거소투표",
    "거소·선상투표",
    "재외투표",
    "국외부재자투표",
    "국외부재자투표(공관)",
    "선상투표",
}
ERROR_LABELS = {"잘못 투입·구분된 투표지"}
SUMMARY_LABEL_NORMALIZATION_MAP = {"계": "합계"}
SPECIAL_DONG_LABELS = ABSENTEE_LABELS | ERROR_LABELS | {"합계", "계"}
SPECIAL_POLLING_LABELS = {"소계", "관내사전투표", "선거일투표"}
CONFIRMED_ELECTORATE_POLLING_LOOKUP_COLUMNS = [
    "선거시점",
    "시도명_F",
    "구시군명_F",
    "일반구명_F",
    "읍면동명_F",
    "투표소명_F",
    "투표소KEY",
]


def _normalize_gusigun_series(series: pd.Series) -> pd.Series:
    cleaned = clean_series(series)
    return cleaned.replace({"화성시갑": "화성시", "화성시을": "화성시", "세종특별자치시": "세종특별자치시"})


def _series_is_absentee(series: pd.Series) -> pd.Series:
    values = clean_series(series).astype("string")
    return values.isin(list(ABSENTEE_LABELS)) | values.str.contains("부재자투표", na=False)


def _series_is_error(series: pd.Series) -> pd.Series:
    values = clean_series(series).astype("string")
    return values.isin(list(ERROR_LABELS)) | values.str.contains("잘못 투입", na=False)


def _normalize_summary_label_series(series: pd.Series) -> pd.Series:
    return clean_series(series).replace(SUMMARY_LABEL_NORMALIZATION_MAP).astype("string")


def _safe_mask(series: pd.Series, condition: pd.Series, value: str) -> pd.Series:
    return series.mask(condition.astype("boolean").fillna(False), value)


def _build_regular_dong_mask(series: pd.Series) -> pd.Series:
    values = _normalize_summary_label_series(series)
    return values.notna() & ~values.isin(list(SPECIAL_DONG_LABELS))


def _build_turnout_rowtype_series(df: pd.DataFrame) -> pd.Series:
    gusigun_name = df["구시군명"].astype("string")
    dong_name = _normalize_summary_label_series(df["읍면동명"])
    polling_name = _normalize_summary_label_series(df["투표구명"])

    rowtype = pd.Series("읍면동", index=df.index, dtype="string")
    rowtype = _safe_mask(rowtype, _series_is_error(dong_name) | _series_is_error(polling_name), "오류투표")
    rowtype = _safe_mask(rowtype, (rowtype == "읍면동") & (gusigun_name.eq("합계") | dong_name.eq("합계")), "합계")
    rowtype = _safe_mask(rowtype, (rowtype == "읍면동") & polling_name.eq("관내사전투표"), "관내사전투표")
    rowtype = _safe_mask(
        rowtype,
        (rowtype == "읍면동") & (_series_is_absentee(dong_name) | _series_is_absentee(polling_name)),
        "관외사전투표",
    )
    rowtype = _safe_mask(
        rowtype,
        (rowtype == "읍면동") & polling_name.notna() & polling_name.ne("") & polling_name.ne("소계"),
        "선거일투표",
    )
    return rowtype


def _build_votes_rowtype_series(df: pd.DataFrame) -> pd.Series:
    sido_name = df["시도명"].astype("string")
    gusigun_name = df["구시군명"].astype("string")
    dong_name = _normalize_summary_label_series(df["읍면동명"])
    division = clean_series(df["구분"]).astype("string")

    rowtype = pd.Series("읍면동", index=df.index, dtype="string")
    rowtype = _safe_mask(rowtype, _series_is_error(dong_name), "오류투표")
    rowtype = _safe_mask(
        rowtype,
        (rowtype == "읍면동") & (sido_name.eq("합계") | gusigun_name.eq("합계") | dong_name.eq("합계")),
        "합계",
    )
    rowtype = _safe_mask(rowtype, (rowtype == "읍면동") & _series_is_absentee(dong_name), "관외사전투표")
    rowtype = _safe_mask(rowtype, (rowtype == "읍면동") & division.eq("관내사전투표"), "관내사전투표")
    rowtype = _safe_mask(rowtype, (rowtype == "읍면동") & division.eq("선거일투표"), "선거일투표")
    rowtype = _safe_mask(
        rowtype,
        (rowtype == "읍면동") & division.notna() & division.ne("") & ~division.isin(list(SPECIAL_POLLING_LABELS)),
        "투표소",
    )
    return rowtype


def _build_confirmed_electorate_rowtype_series(df: pd.DataFrame) -> pd.Series:
    dong_name = _normalize_summary_label_series(df["읍면동명"])
    polling_name = _normalize_summary_label_series(df["투표구명"])

    rowtype = pd.Series("투표소", index=df.index, dtype="string")
    rowtype = _safe_mask(rowtype, dong_name.eq("합계"), "합계")
    rowtype = _safe_mask(rowtype, (rowtype == "투표소") & polling_name.eq("합계"), "읍면동")
    return rowtype


def _normalize_numbered_dong_text_series(series: pd.Series) -> pd.Series:
    values = clean_series(series).astype("string")
    values = values.str.replace(r"제(\d+)(?=동)", r"\1", regex=True)
    values = values.str.replace(r"제(\d+)(?=[·.])", r"\1", regex=True)
    return values.str.replace(r"[·.,]", "·", regex=True)


def _apply_dong_alias(df: pd.DataFrame, dim_dong_alias: pd.DataFrame) -> pd.DataFrame:
    alias_columns = [
        "읍면동KEY_F",
        "시도명_D",
        "구시군명_D",
        "일반구명_D",
        "읍면동명_D",
        "읍면동KEY_D",
    ]
    check_required_columns(dim_dong_alias, alias_columns, "DimDongAlias")

    result = df.merge(dim_dong_alias.loc[:, alias_columns], on="읍면동KEY_F", how="left")
    result["시도명"] = result["시도명_D"].combine_first(result["시도명"])
    result["구시군명"] = result["구시군명_D"].combine_first(result["구시군명"])
    result["일반구명"] = result["일반구명_D"].combine_first(result["일반구명"])
    result["읍면동명"] = result["읍면동명_D"].combine_first(result["읍면동명"])
    return result


def _downcast_turnout_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in FACT_TURNOUT_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("Int32")
    return df


def _downcast_votes_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "후보슬롯" in df.columns:
        df["후보슬롯"] = df["후보슬롯"].astype("Int16")
    for column in ["유효투표수", "득표수"]:
        if column in df.columns:
            df[column] = df[column].astype("Int32")
    if "득표율" in df.columns:
        df["득표율"] = df["득표율"].astype("float32")
    return df


def _downcast_confirmed_electorate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in FACT_CONFIRMED_ELECTORATE_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("Int32")
    return df


def _downcast_resident_composition_numeric(df: pd.DataFrame) -> pd.DataFrame:
    int_columns = {
        "선거연령기준",
        "총인구수",
        "세대수",
        "전월인구수",
        "당월인구수",
        "인구증감",
        "전월남자인구수",
        "전월여자인구수",
        "당월남자인구수",
        "당월여자인구수",
        "남자인구증감",
        "여자인구증감",
        "남자인구수",
        "여자인구수",
        "아동인구",
        "청소년인구",
        "청년인구",
        "고령인구",
        "1인가구수",
        "청년1인가구수_추정",
        "노년1인가구수_추정",
        "10대인구",
        "20대인구",
        "30대인구",
        "40대인구",
        "50대인구",
        "60대인구",
        "70대인구",
        "80대인구",
        "90대인구",
        "100세이상인구",
        "70세이상인구",
        "선거연령청년인구_추정",
        "1인가구연령통계가용여부",
    }
    int_columns.update(FACT_RESIDENT_EXACT_AGE_COLUMNS)
    float_columns = set(FACT_RESIDENT_COMPOSITION_NUMERIC_COLUMNS) - int_columns

    for column in int_columns:
        if column in df.columns:
            df[column] = df[column].astype("Int32")
    if "선거연령기준" in df.columns:
        df["선거연령기준"] = df["선거연령기준"].astype("Int8")
    if "1인가구연령통계가용여부" in df.columns:
        df["1인가구연령통계가용여부"] = df["1인가구연령통계가용여부"].astype("Int8")
    for column in float_columns:
        if column in df.columns:
            df[column] = df[column].astype("float32")
    return df


def _apply_general_gu_from_dim_dong(df: pd.DataFrame, dim_dong: pd.DataFrame | None) -> pd.DataFrame:
    if dim_dong is None or dim_dong.empty:
        return df

    required_columns = {"시도명", "구시군명", "읍면동명", "일반구명"}
    if not required_columns.issubset(dim_dong.columns):
        return df

    dong_lookup = dim_dong.loc[:, ["시도명", "구시군명", "읍면동명", "일반구명"]].copy()
    dong_lookup = dong_lookup.loc[dong_lookup["일반구명"].astype("string").notna()].drop_duplicates(ignore_index=True)
    if dong_lookup.empty:
        return df

    unique_lookup = dong_lookup.groupby(["시도명", "구시군명", "읍면동명"], dropna=False)["일반구명"].nunique(dropna=True)
    unique_lookup = unique_lookup.loc[unique_lookup.eq(1)].reset_index().drop(columns=["일반구명"])
    if unique_lookup.empty:
        return df

    dong_lookup = dong_lookup.merge(unique_lookup, on=["시도명", "구시군명", "읍면동명"], how="inner")
    dong_lookup = dong_lookup.rename(columns={"일반구명": "일반구명_보정"})

    merged = df.merge(
        dong_lookup,
        on=["시도명", "구시군명", "읍면동명"],
        how="left",
        suffixes=("", "_lookup"),
    )
    correction_mask = merged["일반구명_보정"].astype("string").notna()
    merged.loc[correction_mask, "일반구명"] = merged.loc[correction_mask, "일반구명_보정"]
    return merged.drop(columns=["일반구명_보정"], errors="ignore")


def _prepare_confirmed_electorate_polling_lookup(dim_polling_place: pd.DataFrame | None) -> pd.DataFrame:
    if dim_polling_place is None or dim_polling_place.empty:
        return pd.DataFrame()

    required_columns = set(CONFIRMED_ELECTORATE_POLLING_LOOKUP_COLUMNS)
    if not required_columns.issubset(dim_polling_place.columns):
        return pd.DataFrame()

    lookup = dim_polling_place.loc[:, CONFIRMED_ELECTORATE_POLLING_LOOKUP_COLUMNS].copy()
    clean_text_columns(
        lookup,
        ["선거시점", "시도명_F", "구시군명_F", "일반구명_F", "읍면동명_F", "투표소명_F", "투표소KEY"],
        remove_breaks=True,
    )
    lookup["API선거월"] = clean_series(lookup["선거시점"]).str[:6]
    lookup["시도명_매핑"] = normalize_sido_series(lookup["시도명_F"])
    lookup["구시군명_매핑"] = _normalize_gusigun_series(lookup["구시군명_F"])
    lookup["일반구명_매핑"] = clean_series(lookup["일반구명_F"])
    lookup["읍면동명_매핑"] = clean_series(lookup["읍면동명_F"])
    lookup["투표구명_매핑"] = clean_series(lookup["투표소명_F"])

    compact_gugun = compact_location_series(lookup["구시군명_매핑"])
    split_components = compact_gugun.str.extract(r"^(.*?시)(.+구)$")
    split_mask = split_components[1].notna() & lookup["일반구명_매핑"].astype("string").notna()
    lookup.loc[split_mask, "구시군명_매핑"] = split_components.loc[split_mask, 0]
    lookup.loc[split_mask, "일반구명_매핑"] = split_components.loc[split_mask, 1]

    lookup["읍면동명_비교"] = _normalize_numbered_dong_text_series(lookup["읍면동명_매핑"])
    lookup["투표구명_비교"] = _normalize_numbered_dong_text_series(lookup["투표구명_매핑"])

    lookup["시도명_보정"] = lookup["시도명_매핑"]
    lookup["구시군명_보정"] = lookup["구시군명_매핑"]
    lookup["일반구명_보정"] = lookup["일반구명_매핑"]
    lookup["읍면동명_보정"] = lookup["읍면동명_매핑"]
    lookup["투표구명_보정"] = lookup["투표구명_매핑"]
    return lookup.drop_duplicates(ignore_index=True)


def _match_confirmed_electorate_polling_level(
    df: pd.DataFrame,
    pending_mask: pd.Series,
    lookup: pd.DataFrame,
    left_keys: list[str],
    right_key_map: Mapping[str, str],
    *,
    overwrite_sido: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if not pending_mask.any():
        return df, pending_mask

    right_keys = list(right_key_map.keys())
    unique_keys = lookup.groupby(right_keys, dropna=False)["투표소KEY"].nunique(dropna=True).reset_index(name="key_nunique")
    unique_keys = unique_keys.loc[unique_keys["key_nunique"].eq(1), right_keys]
    if unique_keys.empty:
        return df, pending_mask

    candidates = lookup.merge(unique_keys, on=right_keys, how="inner").drop_duplicates(subset=right_keys, ignore_index=True)
    candidates = candidates.rename(columns=right_key_map)
    candidate_columns = list(
        dict.fromkeys(
            left_keys
            + ["투표소KEY", "시도명_보정", "구시군명_보정", "일반구명_보정", "읍면동명_보정", "투표구명_보정"]
        )
    )
    candidates = candidates.loc[:, candidate_columns]

    pending_rows = df.loc[pending_mask, left_keys].copy()
    pending_rows["_row_index"] = pending_rows.index
    matched = pending_rows.merge(candidates, on=left_keys, how="left")
    matched_mask = matched["투표소KEY"].astype("string").notna()
    if not matched_mask.any():
        return df, pending_mask

    updates = matched.loc[matched_mask].set_index("_row_index")
    update_index = updates.index

    if overwrite_sido:
        df.loc[update_index, "시도명"] = updates["시도명_보정"]
    df.loc[update_index, "구시군명"] = updates["구시군명_보정"]
    df.loc[update_index, "일반구명"] = updates["일반구명_보정"]
    df.loc[update_index, "읍면동명"] = updates["읍면동명_보정"]
    df.loc[update_index, "투표구명"] = updates["투표구명_보정"]
    df.loc[update_index, "투표소KEY"] = updates["투표소KEY"]

    remaining_mask = pending_mask.copy()
    remaining_mask.loc[update_index] = False
    return df, remaining_mask


def _apply_confirmed_electorate_polling_lookup(
    df: pd.DataFrame,
    dim_polling_place: pd.DataFrame | None,
) -> pd.DataFrame:
    result = df.copy()
    polling_mask = (
        result["RowType"].eq("투표소")
        & result["투표구명"].astype("string").notna()
        & result["투표구명"].astype("string").ne("합계")
    )
    result["투표소KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if not polling_mask.any():
        return result

    result["API선거월"] = clean_series(result["API선거ID"]).str[:6]
    result["읍면동명_비교"] = _normalize_numbered_dong_text_series(result["읍면동명"])
    result["투표구명_비교"] = _normalize_numbered_dong_text_series(result["투표구명"])
    result.loc[polling_mask, "투표소KEY"] = build_key_series(
        result.loc[polling_mask, "API선거월"],
        result.loc[polling_mask, "시도명"],
        result.loc[polling_mask, "구시군명"],
        result.loc[polling_mask, "일반구명"],
        result.loc[polling_mask, "투표구명"],
    )

    lookup = _prepare_confirmed_electorate_polling_lookup(dim_polling_place)
    if lookup.empty:
        return result.drop(columns=["API선거월", "읍면동명_비교", "투표구명_비교"], errors="ignore")

    valid_polling_keys = set(lookup["투표소KEY"].dropna().astype("string"))
    pending_mask = polling_mask & ~result["투표소KEY"].astype("string").isin(valid_polling_keys)
    if not pending_mask.any():
        return result.drop(columns=["API선거월", "읍면동명_비교", "투표구명_비교"], errors="ignore")

    match_levels = [
        (
            ["API선거월", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명"],
            {
                "API선거월": "API선거월",
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_매핑": "읍면동명",
                "투표구명_매핑": "투표구명",
            },
            True,
        ),
        (
            ["API선거월", "시도명", "구시군명", "읍면동명", "투표구명"],
            {
                "API선거월": "API선거월",
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "읍면동명_매핑": "읍면동명",
                "투표구명_매핑": "투표구명",
            },
            True,
        ),
        (
            ["API선거월", "시도명", "읍면동명", "투표구명"],
            {
                "API선거월": "API선거월",
                "시도명_매핑": "시도명",
                "읍면동명_매핑": "읍면동명",
                "투표구명_매핑": "투표구명",
            },
            True,
        ),
        (
            ["API선거월", "시도명", "구시군명", "일반구명", "읍면동명_비교", "투표구명_비교"],
            {
                "API선거월": "API선거월",
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_비교": "읍면동명_비교",
                "투표구명_비교": "투표구명_비교",
            },
            True,
        ),
        (
            ["API선거월", "구시군명", "일반구명", "읍면동명_비교", "투표구명_비교"],
            {
                "API선거월": "API선거월",
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_비교": "읍면동명_비교",
                "투표구명_비교": "투표구명_비교",
            },
            False,
        ),
    ]

    for left_keys, right_key_map, overwrite_sido in match_levels:
        result, pending_mask = _match_confirmed_electorate_polling_level(
            result,
            pending_mask,
            lookup,
            left_keys,
            right_key_map,
            overwrite_sido=overwrite_sido,
        )
        if not pending_mask.any():
            break

    return result.drop(columns=["API선거월", "읍면동명_비교", "투표구명_비교"], errors="ignore")


def _prepare_confirmed_electorate_dong_lookup(dim_dong: pd.DataFrame | None) -> pd.DataFrame:
    if dim_dong is None or dim_dong.empty:
        return pd.DataFrame()

    required_columns = {"읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"}
    if not required_columns.issubset(dim_dong.columns):
        return pd.DataFrame()

    lookup = (
        dim_dong.loc[:, ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"]]
        .drop_duplicates()
        .rename(
            columns={
                "시도명": "시도명_보정",
                "구시군명": "구시군명_보정",
                "일반구명": "일반구명_보정",
                "읍면동명": "읍면동명_보정",
            }
        )
        .copy()
    )
    lookup["시도명_매핑"] = normalize_sido_series(lookup["시도명_보정"])
    lookup["구시군명_매핑"] = _normalize_gusigun_series(lookup["구시군명_보정"])
    lookup["일반구명_매핑"] = clean_series(lookup["일반구명_보정"])
    lookup["읍면동명_매핑"] = clean_series(lookup["읍면동명_보정"])
    lookup["읍면동명_비교"] = _normalize_numbered_dong_text_series(lookup["읍면동명_매핑"])

    current_gusigun_key = build_key_series(lookup["시도명_매핑"], lookup["구시군명_매핑"])
    lookup["_lookup_priority"] = (~lookup["구시군KEY"].astype("string").eq(current_gusigun_key.astype("string"))).astype("Int8")
    lookup = lookup.sort_values(["_lookup_priority", "읍면동KEY"], kind="stable")
    lookup = lookup.drop_duplicates(
        subset=["시도명_매핑", "구시군명_매핑", "일반구명_매핑", "읍면동명_매핑"],
        keep="first",
    ).reset_index(drop=True)
    return lookup


def _match_confirmed_electorate_dong_level(
    df: pd.DataFrame,
    pending_mask: pd.Series,
    lookup: pd.DataFrame,
    left_keys: list[str],
    right_key_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not pending_mask.any():
        return df, pending_mask

    right_keys = list(right_key_map.keys())
    unique_keys = lookup.groupby(right_keys, dropna=False)["읍면동KEY"].nunique(dropna=True).reset_index(name="key_nunique")
    unique_keys = unique_keys.loc[unique_keys["key_nunique"].eq(1), right_keys]
    if unique_keys.empty:
        return df, pending_mask

    candidates = lookup.merge(unique_keys, on=right_keys, how="inner").drop_duplicates(subset=right_keys, ignore_index=True)
    candidates = candidates.rename(columns=right_key_map)
    candidate_columns = list(
        dict.fromkeys(
            left_keys
            + ["읍면동KEY", "구시군KEY", "시도명_보정", "구시군명_보정", "일반구명_보정", "읍면동명_보정"]
        )
    )
    candidates = candidates.loc[:, candidate_columns]

    pending_rows = df.loc[pending_mask, left_keys].copy()
    pending_rows["_row_index"] = pending_rows.index
    matched = pending_rows.merge(candidates, on=left_keys, how="left")
    matched_mask = matched["읍면동KEY"].astype("string").notna()
    if not matched_mask.any():
        return df, pending_mask

    updates = matched.loc[matched_mask].set_index("_row_index")
    update_index = updates.index
    df.loc[update_index, "시도명"] = updates["시도명_보정"]
    df.loc[update_index, "구시군명"] = updates["구시군명_보정"]
    df.loc[update_index, "일반구명"] = updates["일반구명_보정"]
    df.loc[update_index, "읍면동명"] = updates["읍면동명_보정"]
    df.loc[update_index, "읍면동KEY_보정"] = updates["읍면동KEY"]
    df.loc[update_index, "구시군KEY_보정"] = updates["구시군KEY"]

    remaining_mask = pending_mask.copy()
    remaining_mask.loc[update_index] = False
    return df, remaining_mask


def _apply_confirmed_electorate_dong_lookup(
    df: pd.DataFrame,
    dim_dong: pd.DataFrame | None,
) -> pd.DataFrame:
    result = df.copy()
    relevant_mask = result["RowType"].isin(["읍면동", "투표소"]) & result["읍면동명"].astype("string").notna() & result["읍면동명"].astype("string").ne("합계")
    result["읍면동KEY_보정"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["구시군KEY_보정"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if not relevant_mask.any():
        return result

    result["읍면동명_비교"] = _normalize_numbered_dong_text_series(result["읍면동명"])
    lookup = _prepare_confirmed_electorate_dong_lookup(dim_dong)
    if lookup.empty:
        return result.drop(columns=["읍면동명_비교"], errors="ignore")

    pending_mask = relevant_mask.copy()
    match_levels = [
        (
            ["시도명", "구시군명", "일반구명", "읍면동명"],
            {
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_매핑": "읍면동명",
            },
        ),
        (
            ["시도명", "구시군명", "읍면동명"],
            {
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "읍면동명_매핑": "읍면동명",
            },
        ),
        (
            ["시도명", "구시군명", "일반구명", "읍면동명_비교"],
            {
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_비교": "읍면동명_비교",
            },
        ),
        (
            ["시도명", "구시군명", "읍면동명_비교"],
            {
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
                "읍면동명_비교": "읍면동명_비교",
            },
        ),
        (
            ["시도명", "읍면동명_비교"],
            {
                "시도명_매핑": "시도명",
                "읍면동명_비교": "읍면동명_비교",
            },
        ),
        (
            ["구시군명", "일반구명", "읍면동명_비교"],
            {
                "구시군명_매핑": "구시군명",
                "일반구명_매핑": "일반구명",
                "읍면동명_비교": "읍면동명_비교",
            },
        ),
        (
            ["구시군명", "읍면동명_비교"],
            {
                "구시군명_매핑": "구시군명",
                "읍면동명_비교": "읍면동명_비교",
            },
        ),
    ]

    for left_keys, right_key_map in match_levels:
        result, pending_mask = _match_confirmed_electorate_dong_level(result, pending_mask, lookup, left_keys, right_key_map)
        if not pending_mask.any():
            break

    return result.drop(columns=["읍면동명_비교"], errors="ignore")


def _prepare_confirmed_electorate_gusigun_lookup(dim_gusigun: pd.DataFrame | None) -> pd.DataFrame:
    if dim_gusigun is None or dim_gusigun.empty:
        return pd.DataFrame()

    required_columns = {"구시군KEY", "시도", "구시군"}
    if not required_columns.issubset(dim_gusigun.columns):
        return pd.DataFrame()

    lookup = dim_gusigun.loc[:, ["구시군KEY", "시도", "구시군"]].drop_duplicates().copy()
    lookup["시도명_매핑"] = normalize_sido_series(lookup["시도"])
    lookup["구시군명_매핑"] = _normalize_gusigun_series(lookup["구시군"])

    current_gusigun_key = build_key_series(lookup["시도명_매핑"], lookup["구시군명_매핑"])
    lookup["_lookup_priority"] = (~lookup["구시군KEY"].astype("string").eq(current_gusigun_key.astype("string"))).astype("Int8")
    lookup = lookup.sort_values(["_lookup_priority", "구시군KEY"], kind="stable")
    lookup = lookup.drop_duplicates(subset=["시도명_매핑", "구시군명_매핑"], keep="first").reset_index(drop=True)
    lookup["시도명_보정"] = lookup["시도명_매핑"]
    lookup["구시군명_보정"] = lookup["구시군명_매핑"]
    return lookup


def _match_confirmed_electorate_gusigun_level(
    df: pd.DataFrame,
    pending_mask: pd.Series,
    lookup: pd.DataFrame,
    left_keys: list[str],
    right_key_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not pending_mask.any():
        return df, pending_mask

    right_keys = list(right_key_map.keys())
    unique_keys = lookup.groupby(right_keys, dropna=False)["구시군KEY"].nunique(dropna=True).reset_index(name="key_nunique")
    unique_keys = unique_keys.loc[unique_keys["key_nunique"].eq(1), right_keys]
    if unique_keys.empty:
        return df, pending_mask

    candidates = lookup.merge(unique_keys, on=right_keys, how="inner").drop_duplicates(subset=right_keys, ignore_index=True)
    candidates = candidates.rename(columns=right_key_map)
    candidates = candidates.loc[:, list(dict.fromkeys(left_keys + ["구시군KEY", "시도명_보정", "구시군명_보정"]))]

    pending_rows = df.loc[pending_mask, left_keys].copy()
    pending_rows["_row_index"] = pending_rows.index
    matched = pending_rows.merge(candidates, on=left_keys, how="left")
    matched_mask = matched["구시군KEY"].astype("string").notna()
    if not matched_mask.any():
        return df, pending_mask

    updates = matched.loc[matched_mask].set_index("_row_index")
    update_index = updates.index
    df.loc[update_index, "시도명"] = updates["시도명_보정"]
    df.loc[update_index, "구시군명"] = updates["구시군명_보정"]
    df.loc[update_index, "구시군KEY"] = updates["구시군KEY"]

    remaining_mask = pending_mask.copy()
    remaining_mask.loc[update_index] = False
    return df, remaining_mask


def _apply_confirmed_electorate_gusigun_lookup(
    df: pd.DataFrame,
    dim_gusigun: pd.DataFrame | None,
) -> pd.DataFrame:
    result = df.copy()
    lookup = _prepare_confirmed_electorate_gusigun_lookup(dim_gusigun)
    if lookup.empty:
        return result

    valid_mask = result["구시군명"].astype("string").notna() & result["구시군명"].astype("string").ne("합계")
    if not valid_mask.any():
        return result

    valid_keys = set(lookup["구시군KEY"].dropna().astype("string"))
    pending_mask = valid_mask & ~result["구시군KEY"].astype("string").isin(valid_keys)
    if not pending_mask.any():
        return result

    match_levels = [
        (
            ["시도명", "구시군명"],
            {
                "시도명_매핑": "시도명",
                "구시군명_매핑": "구시군명",
            },
        ),
        (
            ["구시군명"],
            {
                "구시군명_매핑": "구시군명",
            },
        ),
    ]
    for left_keys, right_key_map in match_levels:
        result, pending_mask = _match_confirmed_electorate_gusigun_level(result, pending_mask, lookup, left_keys, right_key_map)
        if not pending_mask.any():
            break

    return result


def transform_fact_resident_composition(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, FACT_RESIDENT_COMPOSITION_RAW_COLUMNS, "FactResidentComposition_Raw")

    result = df.loc[:, FACT_RESIDENT_COMPOSITION_RAW_COLUMNS].copy()
    clean_text_columns(result, FACT_RESIDENT_COMPOSITION_TEXT_COLUMNS, remove_breaks=True)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["시도명_API"] = normalize_sido_series(result["시도명_API"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["구시군명_API"] = _normalize_gusigun_series(result["구시군명_API"])

    for column in FACT_RESIDENT_COMPOSITION_NUMERIC_COLUMNS:
        if column not in result.columns:
            continue
        if column in {
            "세대당인구",
            "남여비율",
            "평균연령",
            "남자평균연령",
            "여자평균연령",
            "인구증감률",
            "남성구성비",
            "여성구성비",
            "선거연령청년구성비_추정",
            "30대구성비",
            "40대구성비",
            "50대구성비",
            "60대구성비",
            "70세이상구성비",
            "아동구성비",
            "청소년구성비",
            "청년구성비",
            "고령구성비",
            "1인가구비율",
            "청년1인가구비율_추정",
            "노년1인가구비율_추정",
        }:
            result[column] = safe_float_cast(result[column])
        else:
            result[column] = safe_int_cast(result[column])

    result = reorder_columns(result, FACT_RESIDENT_COMPOSITION_FINAL_COLUMNS, "FactResidentComposition")
    return _downcast_resident_composition_numeric(result)


def transform_fact_votes_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df_raw, FACT_VOTES_RAW_COLUMNS, "FactVotes_Raw")

    result = df_raw.loc[:, FACT_VOTES_RAW_COLUMNS].copy()
    clean_text_columns(result, FACT_VOTES_TEXT_COLUMNS, remove_breaks=True)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["후보슬롯"] = safe_int_cast(result["후보슬롯"])
    result["유효투표수"] = safe_int_cast(result["유효투표수"])
    result["득표수"] = safe_int_cast(result["득표수"])
    result["득표율"] = safe_float_cast(result["득표율"])
    return _downcast_votes_numeric(reorder_columns(result, FACT_VOTES_RAW_COLUMNS, "FactVotes_Raw"))


def transform_fact_confirmed_electorate(
    df: pd.DataFrame,
    dim_dong_alias: pd.DataFrame,
    dim_dong: pd.DataFrame | None = None,
    dim_polling_place: pd.DataFrame | None = None,
    dim_gusigun: pd.DataFrame | None = None,
) -> pd.DataFrame:
    check_required_columns(df, FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS, "FactConfirmedElectorate_Raw")

    result = df.loc[:, FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS].copy()
    clean_text_columns(result, FACT_CONFIRMED_ELECTORATE_TEXT_COLUMNS, remove_breaks=True)

    result["시도명"] = normalize_sido_series(result["시도명_API"])
    result["구시군명"] = clean_series(result["구시군명_API"])
    result["일반구명"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명_API"])
    result["투표구명"] = _normalize_summary_label_series(result["투표구명_API"])

    compact_gugun = compact_location_series(result["구시군명"])
    split_components = compact_gugun.str.extract(r"^(.*?시)(.+구)$")
    split_mask = split_components[1].notna()
    result.loc[split_mask, "구시군명"] = split_components.loc[split_mask, 0]
    result.loc[split_mask, "일반구명"] = split_components.loc[split_mask, 1]

    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result = _apply_general_gu_from_dim_dong(result, dim_dong)
    for column in FACT_CONFIRMED_ELECTORATE_NUMERIC_COLUMNS:
        result[column] = safe_int_cast(result[column])

    regular_dong_mask = _build_regular_dong_mask(result["읍면동명"])
    result["읍면동KEY_F"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if regular_dong_mask.any():
        result.loc[regular_dong_mask, "읍면동KEY_F"] = build_key_series(
            result.loc[regular_dong_mask, "시도명"],
            result.loc[regular_dong_mask, "구시군명"],
            result.loc[regular_dong_mask, "일반구명"],
            result.loc[regular_dong_mask, "읍면동명"],
        )

    result = _apply_dong_alias(result, dim_dong_alias)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명"])
    result["투표구명"] = _normalize_summary_label_series(result["투표구명"])
    result["RowType"] = _build_confirmed_electorate_rowtype_series(result)
    result = _apply_confirmed_electorate_polling_lookup(result, dim_polling_place)
    result = _apply_confirmed_electorate_dong_lookup(result, dim_dong)

    valid_gugun_mask = result["구시군명"].astype("string").notna() & result["구시군명"].astype("string").ne("합계")
    result["구시군KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if valid_gugun_mask.any():
        result.loc[valid_gugun_mask, "구시군KEY"] = build_key_series(
            result.loc[valid_gugun_mask, "시도명"],
            result.loc[valid_gugun_mask, "구시군명"],
        )
    result["구시군KEY"] = result["구시군KEY_보정"].combine_first(result["구시군KEY"])
    result = _apply_confirmed_electorate_gusigun_lookup(result, dim_gusigun)

    built_dong_key = build_key_series(result["시도명"], result["구시군명"], result["일반구명"], result["읍면동명"])
    dong_key_mask = result["RowType"].isin(["읍면동", "투표소"])
    result["읍면동KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[dong_key_mask, "읍면동KEY"] = result.loc[dong_key_mask, "읍면동KEY_보정"].combine_first(built_dong_key.loc[dong_key_mask])

    result = result.drop(
        columns=[
            "시도명_API",
            "구시군명_API",
            "읍면동명_API",
            "투표구명_API",
            "읍면동KEY_F",
            "시도명_D",
            "구시군명_D",
            "일반구명_D",
            "읍면동명_D",
            "읍면동KEY_D",
            "읍면동KEY_보정",
            "구시군KEY_보정",
        ],
        errors="ignore",
    )
    result = reorder_columns(result, FACT_CONFIRMED_ELECTORATE_FINAL_COLUMNS, "FactConfirmedElectorate")
    return _downcast_confirmed_electorate_numeric(result)


def transform_fact_turnout(df: pd.DataFrame, dim_dong_alias: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, FACT_TURNOUT_COLUMNS, "FactTurnout_Raw")

    result = df.loc[:, FACT_TURNOUT_COLUMNS].copy()
    clean_text_columns(result, FACT_TURNOUT_TEXT_COLUMNS, remove_breaks=True)
    result["선거KEY"] = clean_series(result["선거KEY"]).str.rstrip("-").replace(TURNOUT_KEY_FIXUPS)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명"])
    result["투표구명"] = _normalize_summary_label_series(result["투표구명"])
    for column in FACT_TURNOUT_NUMERIC_COLUMNS:
        result[column] = safe_int_cast(result[column])

    regular_dong_mask = _build_regular_dong_mask(result["읍면동명"])
    result["읍면동KEY_F"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if regular_dong_mask.any():
        result.loc[regular_dong_mask, "읍면동KEY_F"] = build_key_series(
            result.loc[regular_dong_mask, "시도명"],
            result.loc[regular_dong_mask, "구시군명"],
            result.loc[regular_dong_mask, "일반구명"],
            result.loc[regular_dong_mask, "읍면동명"],
        )

    result = _apply_dong_alias(result, dim_dong_alias)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명"])
    result["투표구명"] = _normalize_summary_label_series(result["투표구명"])
    result["RowType"] = _build_turnout_rowtype_series(result)

    valid_gugun_mask = result["구시군명"].astype("string").notna() & result["구시군명"].astype("string").ne("합계")
    result["구시군KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if valid_gugun_mask.any():
        result.loc[valid_gugun_mask, "구시군KEY"] = build_key_series(
            result.loc[valid_gugun_mask, "시도명"],
            result.loc[valid_gugun_mask, "구시군명"],
        )

    built_dong_key = build_key_series(result["시도명"], result["구시군명"], result["일반구명"], result["읍면동명"])
    dong_key_mask = result["RowType"].isin(["읍면동", "관내사전투표", "선거일투표"])
    result["읍면동KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[dong_key_mask, "읍면동KEY"] = result.loc[dong_key_mask, "읍면동KEY_D"].combine_first(
        built_dong_key.loc[dong_key_mask]
    )

    polling_mask = result["투표구명"].astype("string").notna() & result["투표구명"].astype("string").ne("") & result["투표구명"].astype("string").ne("소계")
    result["투표소KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if polling_mask.any():
        result.loc[polling_mask, "투표소KEY"] = build_key_series(
            result.loc[polling_mask, "선거KEY"].astype("string").str[:6],
            result.loc[polling_mask, "시도명"],
            result.loc[polling_mask, "구시군명"],
            result.loc[polling_mask, "일반구명"],
            result.loc[polling_mask, "투표구명"],
        )

    result = result.drop(
        columns=["읍면동KEY_F", "시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D", "읍면동KEY_D"],
        errors="ignore",
    )
    result = reorder_columns(result, FACT_TURNOUT_FINAL_COLUMNS, "FactTurnout")
    return _downcast_turnout_numeric(result)


def transform_fact_votes(
    df: pd.DataFrame,
    dim_party: pd.DataFrame,
    dim_dong_alias: pd.DataFrame,
) -> pd.DataFrame:
    result = transform_fact_votes_raw(df)

    party_lookup = dim_party.loc[:, ["선거KEY", "정당KEY", "정당명"]].drop_duplicates()
    result = result.drop(columns=["정당KEY"], errors="ignore")
    result = result.merge(party_lookup, on=["선거KEY", "정당명"], how="left")

    result["시도명"] = normalize_sido_series(result["시도명"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["구시군명_원본"] = clean_series(result["구시군명"])
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명"])

    compact_gugun = compact_location_series(result["구시군명"])
    split_components = compact_gugun.str.extract(r"^(.*?시)(.+구)$")
    split_mask = split_components[1].notna()

    result["일반구명"] = clean_series(result.get("일반구명", pd.Series(pd.NA, index=result.index)))
    result.loc[split_mask, "구시군명"] = split_components.loc[split_mask, 0]
    result.loc[split_mask, "일반구명"] = split_components.loc[split_mask, 1]

    dong_name = result["읍면동명"].astype("string")
    general_gu_mask = dong_name.str.endswith("구", na=False) & ~dong_name.isin(list(SPECIAL_DONG_LABELS))
    result.loc[general_gu_mask, "일반구명"] = result.loc[general_gu_mask, "읍면동명"]
    result.loc[general_gu_mask, "읍면동명"] = "합계"

    regular_dong_mask = _build_regular_dong_mask(result["읍면동명"])
    result["읍면동KEY_F"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if regular_dong_mask.any():
        result.loc[regular_dong_mask, "읍면동KEY_F"] = build_key_series(
            result.loc[regular_dong_mask, "시도명"],
            result.loc[regular_dong_mask, "구시군명"],
            result.loc[regular_dong_mask, "일반구명"],
            result.loc[regular_dong_mask, "읍면동명"],
        )

    result = _apply_dong_alias(result, dim_dong_alias)
    result["시도명"] = normalize_sido_series(result["시도명"])
    result["구시군명"] = _normalize_gusigun_series(result["구시군명"])
    result["읍면동명"] = _normalize_summary_label_series(result["읍면동명"])
    result["RowType"] = _build_votes_rowtype_series(result)

    valid_gugun_mask = result["구시군명"].astype("string").notna() & result["구시군명"].astype("string").ne("합계")
    result["구시군KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if valid_gugun_mask.any():
        result.loc[valid_gugun_mask, "구시군KEY"] = build_key_series(
            result.loc[valid_gugun_mask, "시도명"],
            result.loc[valid_gugun_mask, "구시군명"],
        )

    built_dong_key = build_key_series(result["시도명"], result["구시군명"], result["일반구명"], result["읍면동명"])
    dong_key_mask = result["RowType"].isin(["읍면동", "관내사전투표", "선거일투표", "투표소"])
    result["읍면동KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[dong_key_mask, "읍면동KEY"] = result.loc[dong_key_mask, "읍면동KEY_D"].combine_first(
        built_dong_key.loc[dong_key_mask]
    )

    polling_mask = result["구분"].astype("string").notna() & result["구분"].astype("string").ne("") & ~result["구분"].astype("string").isin(list(SPECIAL_POLLING_LABELS))
    result["투표소KEY"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if polling_mask.any():
        result.loc[polling_mask, "투표소KEY"] = build_key_series(
            result.loc[polling_mask, "선거KEY"].astype("string").str[:6],
            result.loc[polling_mask, "시도명"],
            result.loc[polling_mask, "구시군명"],
            result.loc[polling_mask, "일반구명"],
            result.loc[polling_mask, "구분"],
        )

    result = result.drop(
        columns=["읍면동KEY_F", "시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D", "읍면동KEY_D"],
        errors="ignore",
    )
    result = reorder_columns(result, FACT_VOTES_FINAL_COLUMNS, "FactVotes")
    return _downcast_votes_numeric(result)


def iter_transform_fact_votes_chunks(
    raw_chunks: Iterable[pd.DataFrame],
    dim_party: pd.DataFrame,
    dim_dong_alias: pd.DataFrame,
) -> Iterator[pd.DataFrame]:
    for raw_chunk in raw_chunks:
        yield transform_fact_votes(raw_chunk, dim_party, dim_dong_alias)


def build_all_facts_from_raw(
    raw_tables: Mapping[str, pd.DataFrame],
    dim_tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    facts: dict[str, pd.DataFrame] = {}

    if "FactResidentComposition_Raw" in raw_tables:
        facts["FactResidentComposition"] = transform_fact_resident_composition(raw_tables["FactResidentComposition_Raw"])
    if "FactConfirmedElectorate_Raw" in raw_tables:
        facts["FactConfirmedElectorate"] = transform_fact_confirmed_electorate(
            raw_tables["FactConfirmedElectorate_Raw"],
            dim_tables["DimDongAlias"],
            dim_tables["DimDong"],
        )
    if "FactTurnout_Raw" in raw_tables:
        facts["FactTurnout"] = transform_fact_turnout(raw_tables["FactTurnout_Raw"], dim_tables["DimDongAlias"])
    if "FactVotes_Raw" in raw_tables:
        facts["FactVotes"] = transform_fact_votes(
            raw_tables["FactVotes_Raw"],
            dim_tables["DimParty"],
            dim_tables["DimDongAlias"],
        )
    return facts


def load_all_facts_from_cache() -> dict[str, pd.DataFrame]:
    from src.loaders import load_cached_facts

    return load_cached_facts()


def build_all_facts(
    raw_tables: Mapping[str, pd.DataFrame] | None = None,
    dim_tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    if raw_tables is None and dim_tables is None:
        return load_all_facts_from_cache()
    if raw_tables is None or dim_tables is None:
        raise ValueError("build_all_facts requires both raw_tables and dim_tables when transforming from raw data.")
    return build_all_facts_from_raw(raw_tables, dim_tables)
