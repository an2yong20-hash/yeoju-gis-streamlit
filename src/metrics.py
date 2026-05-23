from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd

from src.qa import check_required_columns

TURNOUT_NUMERIC_COLUMNS = ["선거인수", "투표수", "유효투표수", "무효투표수", "기권수"]
TURNOUT_RATE_COLUMNS = ["투표율", "유효투표율", "무효투표율", "기권율"]
CONFIRMED_NUMERIC_COLUMNS = ["확정선거인수"]
VOTE_UNIT_COLUMNS = ["선거KEY", "선거구명", "시도명", "구시군명", "일반구명", "읍면동명", "구분", "RowType"]
RANKING_REQUIRED_COLUMNS = ["선거KEY", "정당KEY", "정당명", "후보명", "후보라벨", "득표수"] + VOTE_UNIT_COLUMNS + ["유효투표수"]
ENTITY_GROUP_COLUMN_MAP = {
    "정당": ["정당KEY", "정당명"],
    "후보": ["정당KEY", "정당명", "후보명", "후보라벨"],
    "구분": ["정당구분"],
    "구분2": ["구분2"],
    "성향": ["성향"],
}
ENTITY_LABEL_COLUMN_MAP = {
    "정당": "정당명",
    "후보": "후보명",
    "구분": "정당구분",
    "구분2": "구분2",
    "성향": "성향",
}
GROUP_ENTITY_TYPES = {"구분", "구분2", "성향"}
GROUP_LABEL_ALIASES = {
    "민주당": ("민주당", "더불어민주", "민주통합", "새정치민주"),
    "국민의힘": ("국민의힘", "국민의미래", "새누리당", "자유한국당", "미래통합당"),
    "제3지대": ("제3지대", "국민의당", "조국혁신당", "개혁신당"),
    "진보": ("진보", "진보당", "정의당"),
    "보수": ("보수",),
    "중도/기타": ("중도/기타",),
    "무소속": ("무소속",),
}
GROUP_LABEL_SPECIAL_VALUES = {"기타", "기타 진보", "기타 보수"}
GROUP1_COLLAPSE_LABELS = {"진보", "보수", "중도/기타"}
GROUP2_RENAME_MAP = {"진보": "기타 진보", "보수": "기타 보수"}
ELECTION_META_COLUMNS = ["선거KEY", "선거시점", "선거명", "선거종류"]
ROWTYPE_ORDER = ["합계", "관내사전투표", "관외사전투표", "선거일투표", "투표소", "읍면동", "오류투표"]
ROWTYPE_METHOD_ORDER = ["관내사전투표", "관외사전투표", "선거일투표", "오류투표"]
ROWTYPE_STRUCTURE_ORDER = ["합계", "읍면동", "투표소"]
TURNOUT_COMPONENT_ORDER = ["관내사전투표", "선거일투표", "관외사전투표"]
DONG_LEVEL_EXTRA_ROWTYPES = {"관외사전투표", "오류투표"}
LEVEL_GROUP_COLUMNS = {
    "시도": ["시도명"],
    "구시군": ["구시군KEY", "시도명", "구시군명"],
    "읍면동": ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
    "투표소": ["투표소KEY", "읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
}
LEVEL_LABEL_COLUMNS = {
    "시도": ["시도명"],
    "구시군": ["시도명", "구시군명"],
    "읍면동": ["시도명", "구시군명", "일반구명", "읍면동명"],
    "투표소": ["시도명", "구시군명", "일반구명", "읍면동명"],
}
ELECTION_AXIS_LABEL_MAP = {
    "대통령선거": "대선",
    "지역구": "총선 지역구",
    "비례대표": "총선 비례",
    "광역단체장": "광역단체장",
    "기초단체장": "기초단체장",
    "광역의원비례대표": "광역의원 비례",
    "기초의원비례대표": "기초의원 비례",
    "광역의회": "광역의원",
    "기초의회": "기초의원",
}
BLUE_PARTY_KEYWORDS = ("더불어민주", "민주당", "민주통합", "새정치민주", "더불어민주연합")
RED_PARTY_KEYWORDS = ("국민의힘", "국민의미래", "새누리당", "자유한국당", "미래통합당")


def apply_filters(df: pd.DataFrame, filters: Mapping[str, object] | None) -> pd.DataFrame:
    if not filters:
        return df.copy()

    result = df
    mask = pd.Series(True, index=result.index)

    for column, condition in filters.items():
        if column not in result.columns or condition is None:
            continue

        if callable(condition):
            column_mask = result[column].map(condition).fillna(False)
        elif isinstance(condition, Mapping):
            if "between" in condition:
                lower, upper = condition["between"]
                column_mask = result[column].between(lower, upper)
            elif "isin" in condition:
                column_mask = result[column].isin(list(condition["isin"]))
            elif "eq" in condition:
                column_mask = result[column].eq(condition["eq"])
            else:
                raise ValueError(f"Unsupported filter condition for column '{column}': {condition}")
        elif isinstance(condition, Sequence) and not isinstance(condition, (str, bytes)):
            column_mask = result[column].isin(list(condition))
        else:
            column_mask = result[column].eq(condition)

        mask = mask & column_mask.fillna(False)

    return result.loc[mask].copy()


def safe_divide(a: object, b: object) -> object:
    if np.isscalar(a) and np.isscalar(b):
        if pd.isna(a) or pd.isna(b) or b in (0, 0.0):
            return np.nan
        return float(a) / float(b)

    numerator = a if isinstance(a, pd.Series) else pd.Series(a)
    denominator = b if isinstance(b, pd.Series) else pd.Series(b, index=numerator.index)
    denominator = denominator.astype("float64")
    result = numerator.astype("float64").divide(denominator)
    return result.mask(denominator.isna() | denominator.eq(0))


def _major_party_gap_sign(party_name: object) -> int:
    if pd.isna(party_name):
        return 0
    value = str(party_name)
    if any(keyword in value for keyword in BLUE_PARTY_KEYWORDS):
        return -1
    if any(keyword in value for keyword in RED_PARTY_KEYWORDS):
        return 1
    return 0


def _gap_direction_label(party_name: object) -> str:
    sign = _major_party_gap_sign(party_name)
    if sign < 0:
        return "민주계 우세"
    if sign > 0:
        return "보수계 우세"
    return "기타 정당 우세"


def format_percent(x: object) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:.2f}%"


def format_int(x: object) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(round(float(x))):,}"


def format_delta_pp(x: object) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:+.2f}%p"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_region_label(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.Series:
    region_columns = [
        column
        for column in (columns or ["시도명", "구시군명", "일반구명", "읍면동명"])
        if column in df.columns and not column.endswith("KEY")
    ]
    if not region_columns:
        return pd.Series(["지역"] * len(df), index=df.index, dtype="string")

    label = df[region_columns[0]].astype("string").fillna("")
    for column in region_columns[1:]:
        label = label.str.cat(df[column].astype("string").fillna(""), sep=" ")
    return label.str.replace(r"\s+", " ", regex=True).str.strip().replace("", "지역")


def _normalize_election_axis_kind(values: pd.Series) -> pd.Series:
    normalized = values.astype("string")
    for source, target in ELECTION_AXIS_LABEL_MAP.items():
        normalized = normalized.mask(normalized.eq(source), target)
    return normalized


def _election_year_series(df: pd.DataFrame) -> pd.Series:
    if "선거시점" in df.columns:
        year = df["선거시점"].astype("string").str.slice(0, 4)
        if year.notna().any():
            return year
    if "선거KEY" in df.columns:
        return df["선거KEY"].astype("string").str.slice(0, 4)
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def build_election_label(df: pd.DataFrame) -> pd.Series:
    year = _election_year_series(df)
    if "선거종류" in df.columns:
        axis_kind = _normalize_election_axis_kind(df["선거종류"])
        full_name = df["선거명"].astype("string") if "선거명" in df.columns else axis_kind
        return year.str.cat(axis_kind, sep=" ").str.strip().str.cat(full_name, sep=" | ")
    if {"선거시점", "선거명"}.issubset(df.columns):
        return year.str.cat(df["선거명"].astype("string"), sep=" | ")
    if "선거KEY" in df.columns:
        return df["선거KEY"].astype("string")
    return pd.Series(["선거"] * len(df), index=df.index, dtype="string")


def build_election_axis_label(df: pd.DataFrame) -> pd.Series:
    year = _election_year_series(df)
    if "선거종류" in df.columns:
        axis_kind = _normalize_election_axis_kind(df["선거종류"])
        return year.str.cat(axis_kind, sep=" ").str.strip()
    if "선거명" in df.columns:
        return year.str.cat(df["선거명"].astype("string"), sep=" ").str.strip()
    if "선거KEY" in df.columns:
        return year.str.cat(df["선거KEY"].astype("string"), sep=" ").str.strip()
    return build_election_label(df)


def _attach_election_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["선거라벨"] = build_election_label(result)
    result["선거축라벨"] = build_election_axis_label(result)
    return result


def get_latest_and_previous(df: pd.DataFrame, value_col: str) -> dict[str, object]:
    if df.empty or value_col not in df.columns:
        return {"current": np.nan, "previous": np.nan, "delta": np.nan, "latest_key": None}

    sort_columns = [column for column in ["선거시점", "선거KEY"] if column in df.columns]
    ordered = df.sort_values(by=sort_columns or [value_col], ascending=[False] * max(len(sort_columns), 1), kind="stable")
    latest = ordered.iloc[0]
    previous = ordered.iloc[1] if len(ordered) > 1 else None
    current_value = latest[value_col]
    previous_value = np.nan if previous is None else previous[value_col]
    return {
        "current": current_value,
        "previous": previous_value,
        "delta": np.nan if previous is None else current_value - previous_value,
        "latest_key": latest.get("선거KEY"),
    }


def build_delta_sentence(subject: str, metric_name: str, delta: object) -> str:
    if pd.isna(delta):
        return f"{subject}의 {metric_name} 비교 기준이 부족합니다."

    delta_value = float(delta)
    if abs(delta_value) < 1e-12:
        return f"{subject}의 {metric_name}은 직전 선거와 같은 수준입니다."

    direction = "상승" if delta_value > 0 else "하락"
    return f"{subject}의 {metric_name}은 직전 선거 대비 {abs(delta_value) * 100:.1f}%p {direction}했습니다."


def build_vote_rowtype_sentence(rowtype_df: pd.DataFrame, entity_label: str) -> str:
    if rowtype_df.empty or "득표율_계산" not in rowtype_df.columns:
        return f"{entity_label}의 RowType별 비교 기준이 부족합니다."

    base = rowtype_df.loc[rowtype_df["RowType"].isin(["관내사전투표", "선거일투표"])].copy()
    if len(base) < 2:
        return f"{entity_label}의 주요 RowType 비교 기준이 부족합니다."

    base = base.sort_values(by="득표율_계산", ascending=False, kind="stable").reset_index(drop=True)
    first = base.iloc[0]
    second = base.iloc[1]
    gap = float(first["득표율_계산"] - second["득표율_계산"])
    return f"{entity_label}는 {first['RowType']}에서 더 강하며, {second['RowType']} 대비 {abs(gap) * 100:.1f}%p 앞섭니다."


def build_competitiveness_sentence(gap_df: pd.DataFrame, subject: str) -> str:
    if gap_df.empty or "득표율격차" not in gap_df.columns:
        return f"{subject}의 경쟁도 해석 기준이 부족합니다."

    gap = gap_df.iloc[0]["득표율격차"]
    if pd.isna(gap):
        return f"{subject}는 단일 후보 집계로 경쟁도 비교가 어렵습니다."
    if float(gap) <= 0.05:
        return f"{subject}는 1위-2위 격차가 {float(gap) * 100:.1f}%p로 매우 접전입니다."
    return f"{subject}는 1위-2위 격차가 {float(gap) * 100:.1f}%p로 비교적 여유 있는 구도입니다."


def competitiveness_help_text() -> str:
    # 경쟁도지수는 1위와 2위의 득표율 차이를 뒤집어 만든 접전 지표다.
    return "경쟁도지수 = 1 - 1위·2위 득표율 격차입니다. 1에 가까울수록 접전이고, 0에 가까울수록 격차가 큽니다."


def _attach_election_metadata(grouped: pd.DataFrame, source_df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    if dim_election is not None:
        meta = dim_election.loc[:, ELECTION_META_COLUMNS].drop_duplicates(subset=["선거KEY"])
        return grouped.merge(meta, on="선거KEY", how="left", copy=False)

    if all(column in source_df.columns for column in ELECTION_META_COLUMNS):
        meta = source_df.loc[:, ELECTION_META_COLUMNS].drop_duplicates(subset=["선거KEY"])
        return grouped.merge(meta, on="선거KEY", how="left", copy=False)

    return grouped


def _is_concrete_hierarchy_value(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    return values.notna() & values.ne("") & ~values.eq("전국") & ~values.str.startswith("합계", na=False)


def _summary_specificity_score(df: pd.DataFrame, weighted_columns: Mapping[str, int]) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype="int16")
    for column, weight in weighted_columns.items():
        if column not in df.columns:
            continue
        score = score + _is_concrete_hierarchy_value(df[column]).astype("int16") * int(weight)
    return score


def _select_hierarchical_summary_rows(
    df: pd.DataFrame,
    weighted_columns: Mapping[str, int],
    fallback_mask: pd.Series,
    identity_columns: Sequence[str],
    prefer_summary_when_present: bool = False,
) -> pd.DataFrame:
    check_required_columns(df, ["선거KEY", "RowType"], "summary frame")
    del identity_columns

    selected_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("선거KEY", observed=True, sort=False):
        summary_rows = group.loc[group["RowType"].eq("합계")].copy()
        if not summary_rows.empty:
            specificity = _summary_specificity_score(summary_rows, weighted_columns)
            max_specificity = int(specificity.max())
            if max_specificity > 0:
                selected_parts.append(summary_rows.loc[specificity.eq(max_specificity)])
                continue
            if prefer_summary_when_present:
                selected_parts.append(summary_rows)
                continue

        fallback_rows = group.loc[fallback_mask.loc[group.index]]
        if not fallback_rows.empty:
            selected_parts.append(fallback_rows)
        elif not summary_rows.empty:
            selected_parts.append(summary_rows)

    if not selected_parts:
        return df.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=True)


def _select_turnout_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, ["RowType", "시도명", "구시군명", "읍면동명"], "turnout frame")
    fallback_mask = df["RowType"].eq("읍면동")
    return _select_hierarchical_summary_rows(
        df,
        weighted_columns={"시도명": 1, "구시군명": 2},
        fallback_mask=fallback_mask,
        identity_columns=TURNOUT_NUMERIC_COLUMNS,
    )


def _normalize_level(level: str) -> str:
    normalized_level = level.strip()
    if normalized_level not in LEVEL_GROUP_COLUMNS:
        raise ValueError(f"Unsupported level: {level}")
    return normalized_level


def _has_group_values(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series.dtype):
        values = series.astype("string").str.strip().replace("", pd.NA)
        return values.notna().any()
    return series.notna().any()


def _resolve_level_group_columns(df: pd.DataFrame, level: str) -> list[str]:
    normalized_level = _normalize_level(level)
    return [
        column
        for column in LEVEL_GROUP_COLUMNS[normalized_level]
        if column in df.columns and _has_group_values(df[column])
    ]


def _select_turnout_rows_by_level(df: pd.DataFrame, level: str) -> tuple[pd.DataFrame, list[str]]:
    normalized_level = _normalize_level(level)
    group_cols = _resolve_level_group_columns(df, normalized_level)
    dong = df["읍면동명"].astype("string") if "읍면동명" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    if normalized_level == "시도":
        return df.loc[df["RowType"].eq("합계") & dong.eq("합계")], group_cols
    if normalized_level == "구시군":
        return df.loc[df["RowType"].eq("합계") & dong.eq("합계")], group_cols
    if normalized_level == "읍면동":
        rowtype = df["RowType"].astype("string")
        return df.loc[rowtype.eq("읍면동") | rowtype.isin(DONG_LEVEL_EXTRA_ROWTYPES)], group_cols
    return df.loc[df["투표소KEY"].notna()], group_cols


def _select_confirmed_base_rows(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, ["선거KEY", "RowType", "확정선거인수"], "confirmed electorate frame")

    selected_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("선거KEY", observed=True, sort=False):
        for rowtype in ("투표소", "읍면동", "합계"):
            matched = group.loc[group["RowType"].eq(rowtype)].copy()
            if not matched.empty:
                selected_parts.append(matched)
                break

    if not selected_parts:
        return df.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=True)


def _select_confirmed_rows_by_level(df: pd.DataFrame, level: str) -> tuple[pd.DataFrame, list[str]]:
    normalized_level = _normalize_level(level)
    base = _select_confirmed_base_rows(df)
    group_cols = _resolve_level_group_columns(base if not base.empty else df, normalized_level)
    if base.empty:
        return base, group_cols

    if normalized_level == "시도":
        return base, group_cols
    if normalized_level == "구시군":
        return base.loc[base["구시군KEY"].notna()].copy(), group_cols
    if normalized_level == "읍면동":
        return base.loc[base["읍면동KEY"].notna()].copy(), group_cols
    return base.loc[base["투표소KEY"].notna()].copy(), group_cols


def _select_vote_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, ["RowType", "시도명", "구시군명", "읍면동명"], "votes frame")
    fallback_mask = df["RowType"].eq("읍면동")
    return _select_hierarchical_summary_rows(
        df,
        weighted_columns={"선거구명": 4, "시도명": 1, "구시군명": 2},
        fallback_mask=fallback_mask,
        identity_columns=["정당KEY", "정당명", "후보명", "후보라벨", "유효투표수", "득표수"],
        prefer_summary_when_present=True,
    )


def _select_vote_rows_by_level(df: pd.DataFrame, level: str) -> tuple[pd.DataFrame, list[str]]:
    normalized_level = _normalize_level(level)
    group_cols = _resolve_level_group_columns(df, normalized_level)
    dong = df["읍면동명"].astype("string") if "읍면동명" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    if normalized_level == "시도":
        return df.loc[df["RowType"].eq("합계") & dong.eq("합계")], group_cols
    if normalized_level == "구시군":
        return df.loc[df["RowType"].eq("합계") & dong.eq("합계")], group_cols
    if normalized_level == "읍면동":
        rowtype = df["RowType"].astype("string")
        return df.loc[rowtype.eq("읍면동") | rowtype.isin(DONG_LEVEL_EXTRA_ROWTYPES)], group_cols
    return df.loc[df["투표소KEY"].notna()], group_cols


def _add_turnout_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["투표율"] = safe_divide(result["투표수"], result["선거인수"])
    result["유효투표율"] = safe_divide(result["유효투표수"], result["투표수"])
    result["무효투표율"] = safe_divide(result["무효투표수"], result["투표수"])
    result["기권율"] = safe_divide(result["기권수"], result["선거인수"])
    return result


def _sum_unique_vote_units(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    check_required_columns(df, VOTE_UNIT_COLUMNS + ["유효투표수"], "vote unit frame")

    extra_group_cols = [column for column in (group_cols or []) if column not in VOTE_UNIT_COLUMNS and column in df.columns]
    projected_columns = [*VOTE_UNIT_COLUMNS, *extra_group_cols, "유효투표수"]
    unique_units = df.loc[:, projected_columns].drop_duplicates(subset=projected_columns)
    if group_cols:
        return unique_units.groupby(group_cols, dropna=False, as_index=False, observed=True)["유효투표수"].sum(min_count=1)
    return pd.DataFrame({"유효투표수": [unique_units["유효투표수"].sum(min_count=1)]})


def _grouped_unique_count(df: pd.DataFrame, group_cols: list[str], identity_cols: list[str], result_name: str) -> pd.DataFrame:
    available_identity_cols = [column for column in identity_cols if column in df.columns]
    if not available_identity_cols:
        return pd.DataFrame(columns=[*group_cols, result_name])

    unique_entities = df.loc[:, [*group_cols, *available_identity_cols]].dropna(
        subset=available_identity_cols,
        how="all",
    ).drop_duplicates()
    return unique_entities.groupby(group_cols, as_index=False, observed=True).size().rename(columns={"size": result_name})


def _unique_entity_count(df: pd.DataFrame, identity_cols: list[str]) -> int:
    available_identity_cols = [column for column in identity_cols if column in df.columns]
    if not available_identity_cols:
        return 0
    unique_entities = df.loc[:, available_identity_cols].dropna(subset=available_identity_cols, how="all").drop_duplicates()
    return int(len(unique_entities))


def _entity_group_columns(entity_type: str) -> list[str]:
    if entity_type not in ENTITY_GROUP_COLUMN_MAP:
        raise ValueError(f"Unsupported entity type: {entity_type}")
    return list(ENTITY_GROUP_COLUMN_MAP[entity_type])


def _entity_label_column(entity_type: str) -> str:
    if entity_type not in ENTITY_LABEL_COLUMN_MAP:
        raise ValueError(f"Unsupported entity type: {entity_type}")
    return ENTITY_LABEL_COLUMN_MAP[entity_type]


def _normalize_group_label(value: object, entity_type: str) -> object:
    if pd.isna(value):
        return value
    label = re.sub(r"^\s*\d+\.\s*", "", str(value)).strip()
    if label in GROUP_LABEL_SPECIAL_VALUES:
        return label

    normalized_label = label
    for canonical, aliases in GROUP_LABEL_ALIASES.items():
        if any(alias in label for alias in aliases):
            normalized_label = canonical
            break

    if entity_type == "구분" and normalized_label in GROUP1_COLLAPSE_LABELS:
        return "기타"
    if entity_type == "구분2":
        return GROUP2_RENAME_MAP.get(normalized_label, normalized_label)
    return normalized_label


def _normalize_entity_labels(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if df.empty or entity_type not in GROUP_ENTITY_TYPES:
        return df
    label_col = _entity_label_column(entity_type)
    if label_col not in df.columns:
        return df
    normalized = df.copy()
    normalized[label_col] = normalized[label_col].map(lambda value: _normalize_group_label(value, entity_type))
    return normalized


def _entity_required_columns(entity_type: str) -> list[str]:
    required = ["선거KEY", "득표수", "유효투표수", *VOTE_UNIT_COLUMNS]
    for column in _entity_group_columns(entity_type):
        if column not in required:
            required.append(column)
    return required


def _add_period_change(df: pd.DataFrame, value_cols: Sequence[str], group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    sort_columns = [column for column in ["선거시점", "선거KEY"] if column in result.columns]
    group_cols = list(group_cols or [])
    result = result.sort_values(by=[*group_cols, *sort_columns], kind="stable").reset_index(drop=True)

    for value_col in value_cols:
        if value_col not in result.columns:
            continue
        prev_col = f"{value_col}_이전"
        delta_col = f"{value_col}_증감"
        if group_cols:
            result[prev_col] = result.groupby(group_cols, observed=True)[value_col].shift(1)
        else:
            result[prev_col] = result[value_col].shift(1)
        result[delta_col] = result[value_col] - result[prev_col]

    return result


def _sort_rowtypes(df: pd.DataFrame) -> pd.DataFrame:
    if "RowType" not in df.columns:
        return df
    rowtype_order = {name: idx for idx, name in enumerate(ROWTYPE_ORDER)}
    result = df.copy()
    result["_rowtype_order"] = result["RowType"].map(rowtype_order).fillna(len(ROWTYPE_ORDER))
    result = result.sort_values(by=["_rowtype_order", "RowType"], kind="stable").drop(columns="_rowtype_order")
    return result.reset_index(drop=True)


def _sort_turnout_components(df: pd.DataFrame) -> pd.DataFrame:
    if "RowType" not in df.columns:
        return df
    rowtype_order = {"전체 합계": -1, **{name: idx for idx, name in enumerate(TURNOUT_COMPONENT_ORDER)}}
    result = df.copy()
    result["_turnout_component_order"] = result["RowType"].map(rowtype_order).fillna(len(TURNOUT_COMPONENT_ORDER))
    result = result.sort_values(by=["_turnout_component_order", "RowType"], kind="stable").drop(columns="_turnout_component_order")
    return result.reset_index(drop=True)


def _is_outside_turnout_label(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("")
    return values.str.contains(r"관외|거소|선상|재외|부재", na=False)


def _is_summary_polling_key(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("")
    return values.str.contains(r"합계", na=False)


def _select_direct_day_rows(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df["RowType"].isin(["선거일투표", "투표소"])].copy()
    if base.empty:
        return base
    if "투표소KEY" not in base.columns:
        return base
    summary_mask = _is_summary_polling_key(base["투표소KEY"])
    if "구분" in base.columns:
        summary_mask = summary_mask | base["구분"].astype("string").eq("합계")
    if (~summary_mask).any():
        return base.loc[~summary_mask].copy()
    return base


def _aggregate_turnout_components(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    check_required_columns(df, ["RowType", "읍면동명", "투표수"], "turnout component frame")

    available_group_cols = [column for column in group_cols if column in df.columns]
    available_numeric_cols = [column for column in TURNOUT_NUMERIC_COLUMNS if column in df.columns]
    rowtype = df["RowType"].astype("string")
    local_total = df.loc[rowtype.eq("읍면동")].copy()
    early_base = df.loc[rowtype.eq("관내사전투표")].copy()
    outside_base = df.loc[rowtype.eq("관외사전투표")].copy()
    direct_day = _select_direct_day_rows(df)
    if local_total.empty and early_base.empty and outside_base.empty and direct_day.empty:
        return pd.DataFrame(columns=[*available_group_cols, "RowType", *TURNOUT_NUMERIC_COLUMNS, "투표구성비"])

    outside_mask = _is_outside_turnout_label(early_base["읍면동명"]) if not early_base.empty else pd.Series(dtype=bool)
    local_early = early_base.loc[~outside_mask].copy()
    legacy_outside = early_base.loc[outside_mask].copy()
    outside_parts = [frame for frame in [outside_base, legacy_outside] if not frame.empty]
    outside = pd.concat(outside_parts, ignore_index=True) if outside_parts else outside_base

    def _group_numeric(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[*available_group_cols, *[f"{prefix}_{column}" for column in available_numeric_cols]])
        if available_group_cols:
            grouped = frame.groupby(available_group_cols, as_index=False, observed=True)[available_numeric_cols].sum(min_count=1)
        else:
            grouped = pd.DataFrame([frame[available_numeric_cols].sum(min_count=1).to_dict()])
        return grouped.rename(columns={column: f"{prefix}_{column}" for column in available_numeric_cols})

    if available_group_cols:
        key_frames = [frame.loc[:, available_group_cols] for frame in [local_total, local_early, outside, direct_day] if not frame.empty]
        if not key_frames:
            return pd.DataFrame(columns=[*available_group_cols, "RowType", *TURNOUT_NUMERIC_COLUMNS, "투표구성비"])
        key_frame = pd.concat(key_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    else:
        key_frame = pd.DataFrame([{}])
    component_base = key_frame.copy()
    if available_group_cols:
        component_base = component_base.merge(_group_numeric(local_total, "local_total"), on=available_group_cols, how="left", copy=False)
        component_base = component_base.merge(_group_numeric(local_early, "local_early"), on=available_group_cols, how="left", copy=False)
        component_base = component_base.merge(_group_numeric(outside, "outside"), on=available_group_cols, how="left", copy=False)
        component_base = component_base.merge(_group_numeric(direct_day, "direct_day"), on=available_group_cols, how="left", copy=False)
    else:
        for grouped_frame in [_group_numeric(local_total, "local_total"), _group_numeric(local_early, "local_early"), _group_numeric(outside, "outside"), _group_numeric(direct_day, "direct_day")]:
            if grouped_frame.empty:
                continue
            component_base = pd.concat([component_base.reset_index(drop=True), grouped_frame.reset_index(drop=True)], axis=1)

    for prefix in ["local_total", "local_early", "outside", "direct_day"]:
        for column in available_numeric_cols:
            prefixed_column = f"{prefix}_{column}"
            if prefixed_column not in component_base.columns:
                component_base[prefixed_column] = pd.NA

    local_total_available = component_base["local_total_투표수"].notna()
    direct_day_available = component_base["direct_day_투표수"].notna()

    for column in available_numeric_cols:
        local_total_col = component_base[f"local_total_{column}"].fillna(0)
        local_early_col = component_base[f"local_early_{column}"].fillna(0)
        outside_col = component_base[f"outside_{column}"].fillna(0)
        direct_day_col = component_base[f"direct_day_{column}"].fillna(0)
        component_base[f"관내사전투표_{column}"] = local_early_col
        election_day_col = (local_total_col - local_early_col).clip(lower=0)
        election_day_col = election_day_col.where(local_total_available, direct_day_col.where(direct_day_available, 0))
        component_base[f"선거일투표_{column}"] = election_day_col
        component_base[f"관외사전투표_{column}"] = outside_col

    result_parts: list[pd.DataFrame] = []
    for rowtype in TURNOUT_COMPONENT_ORDER:
        part = component_base.loc[:, available_group_cols].copy()
        part["RowType"] = rowtype
        for column in TURNOUT_NUMERIC_COLUMNS:
            part[column] = component_base[f"{rowtype}_{column}"] if column in available_numeric_cols else pd.NA
        result_parts.append(part)

    result = pd.concat(result_parts, ignore_index=True)
    if available_group_cols:
        totals = result.groupby(available_group_cols, as_index=False, observed=True)["투표수"].sum(min_count=1).rename(columns={"투표수": "총투표수"})
        result = result.merge(totals, on=available_group_cols, how="left", copy=False)
    else:
        result["총투표수"] = result["투표수"].sum(min_count=1)
    result = result.loc[result["총투표수"].fillna(0).gt(0)].copy()
    result["투표구성비"] = safe_divide(result["투표수"], result["총투표수"])
    result = result.drop(columns="총투표수")
    return _sort_turnout_components(result)


def calc_turnout_summary(df: pd.DataFrame) -> dict[str, object]:
    base = _select_turnout_summary_rows(df)
    if base.empty:
        return {
            "rows_used": 0,
            "선거인수": 0,
            "투표수": 0,
            "유효투표수": 0,
            "무효투표수": 0,
            "기권수": 0,
            "투표율": np.nan,
            "유효투표율": np.nan,
            "무효투표율": np.nan,
            "기권율": np.nan,
        }

    totals = base[TURNOUT_NUMERIC_COLUMNS].astype("Int64").sum(min_count=1)
    summary = {column: totals[column] for column in TURNOUT_NUMERIC_COLUMNS}
    summary["rows_used"] = int(len(base))
    summary["투표율"] = safe_divide(summary["투표수"], summary["선거인수"])
    summary["유효투표율"] = safe_divide(summary["유효투표수"], summary["투표수"])
    summary["무효투표율"] = safe_divide(summary["무효투표수"], summary["투표수"])
    summary["기권율"] = safe_divide(summary["기권수"], summary["선거인수"])
    return summary


def calc_confirmed_electorate_summary(df: pd.DataFrame) -> dict[str, object]:
    base = _select_confirmed_base_rows(df)
    if base.empty:
        return {
            "rows_used": 0,
            "확정선거인수": 0,
        }

    return {
        "rows_used": int(len(base)),
        "확정선거인수": base["확정선거인수"].astype("Int64").sum(min_count=1),
    }


def calc_turnout_by_region(df: pd.DataFrame, level: str = "구시군") -> pd.DataFrame:
    base, group_cols = _select_turnout_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(columns=group_cols + TURNOUT_NUMERIC_COLUMNS + TURNOUT_RATE_COLUMNS)

    grouped = base.groupby(group_cols, dropna=False, as_index=False, observed=True)[TURNOUT_NUMERIC_COLUMNS].sum(min_count=1)
    grouped = _add_turnout_rates(grouped)
    grouped["지역"] = build_region_label(grouped, group_cols)
    return grouped.sort_values(by=["투표율", "투표수"], ascending=[False, False], kind="stable").reset_index(drop=True)


def calc_confirmed_electorate_by_region(df: pd.DataFrame, level: str = "구시군") -> pd.DataFrame:
    base, group_cols = _select_confirmed_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(columns=[*group_cols, "확정선거인수", "지역"])

    grouped = base.groupby(group_cols, dropna=False, as_index=False, observed=True)["확정선거인수"].sum(min_count=1)
    grouped["지역"] = build_region_label(grouped, group_cols)
    return grouped.sort_values(by=["확정선거인수", "지역"], ascending=[False, True], kind="stable").reset_index(drop=True)


def calc_turnout_by_rowtype(df: pd.DataFrame) -> pd.DataFrame:
    grouped = _aggregate_turnout_components(df, [])
    if grouped.empty:
        return pd.DataFrame(columns=["RowType", *TURNOUT_NUMERIC_COLUMNS, "투표구성비"])
    return _sort_turnout_components(grouped)


def calc_turnout_timeseries(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    summary_rows = _select_turnout_summary_rows(df)
    if summary_rows.empty:
        return pd.DataFrame(columns=ELECTION_META_COLUMNS + TURNOUT_NUMERIC_COLUMNS + TURNOUT_RATE_COLUMNS)

    grouped = summary_rows.groupby("선거KEY", as_index=False, observed=True)[TURNOUT_NUMERIC_COLUMNS].sum(min_count=1)
    grouped = _add_turnout_rates(grouped)
    grouped = _attach_election_metadata(grouped, summary_rows, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return _add_period_change(grouped, ["투표율", "유효투표율", "무효투표율", "기권율"])


def calc_turnout_summary_by_election(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    base = _select_turnout_summary_rows(df)
    columns = ["선거KEY"] + TURNOUT_NUMERIC_COLUMNS + TURNOUT_RATE_COLUMNS
    if base.empty:
        return pd.DataFrame(columns=columns + ELECTION_META_COLUMNS[1:])

    grouped = (
        base.groupby("선거KEY", as_index=False, observed=True)[TURNOUT_NUMERIC_COLUMNS]
        .sum(min_count=1)
        .astype({column: "Int64" for column in TURNOUT_NUMERIC_COLUMNS})
    )
    grouped = _add_turnout_rates(grouped)
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return grouped


def calc_confirmed_electorate_summary_by_election(
    df: pd.DataFrame,
    dim_election: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = _select_confirmed_base_rows(df)
    columns = ["선거KEY", "확정선거인수"]
    if base.empty:
        return pd.DataFrame(columns=columns + ELECTION_META_COLUMNS[1:])

    grouped = (
        base.groupby("선거KEY", as_index=False, observed=True)["확정선거인수"]
        .sum(min_count=1)
        .astype({"확정선거인수": "Int64"})
    )
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return grouped


def calc_turnout_trend_by_level(df: pd.DataFrame, level: str = "구시군", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    base, region_cols = _select_turnout_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(columns=["선거KEY", *region_cols, *TURNOUT_NUMERIC_COLUMNS, *TURNOUT_RATE_COLUMNS, "지역", "선거라벨"])

    grouped = base.groupby(["선거KEY", *region_cols], as_index=False, observed=True)[TURNOUT_NUMERIC_COLUMNS].sum(min_count=1)
    grouped = _add_turnout_rates(grouped)
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped["지역"] = build_region_label(grouped, region_cols)
    grouped = _attach_election_label_columns(grouped)
    return _add_period_change(grouped, ["투표율", "유효투표율", "무효투표율"], ["지역"])


def calc_confirmed_electorate_trend_by_level(
    df: pd.DataFrame,
    level: str = "구시군",
    dim_election: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base, region_cols = _select_confirmed_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(columns=["선거KEY", *region_cols, "확정선거인수", "지역", "선거라벨"])

    grouped = base.groupby(["선거KEY", *region_cols], as_index=False, observed=True)["확정선거인수"].sum(min_count=1)
    grouped["지역"] = build_region_label(grouped, region_cols)
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return _add_period_change(grouped, ["확정선거인수"], ["지역"])


def calc_rowtype_turnout_trend(df: pd.DataFrame, family: str = "투표방식", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    del family
    grouped = _aggregate_turnout_components(df, ["선거KEY"])
    if grouped.empty:
        return pd.DataFrame(columns=["선거KEY", "RowType", "투표수", "투표구성비"] + ELECTION_META_COLUMNS[1:])
    grouped = _attach_election_metadata(grouped, df, dim_election)
    grouped = _attach_election_label_columns(grouped)
    grouped = _add_period_change(grouped, ["투표구성비"], ["RowType"])
    return _sort_turnout_components(grouped)


def calc_turnout_rowtype_mix_by_region(df: pd.DataFrame, level: str = "구시군") -> pd.DataFrame:
    _, region_cols = _select_turnout_rows_by_level(df, level)
    mix = _aggregate_turnout_components(df, region_cols)
    if mix.empty:
        return pd.DataFrame(columns=[*region_cols, "RowType", "투표수", "투표구성비", "지역"])
    available_region_cols = [column for column in region_cols if column in mix.columns]
    mix["지역"] = build_region_label(mix, available_region_cols)
    return _sort_turnout_components(mix)


def calc_votes_summary(df: pd.DataFrame) -> dict[str, object]:
    base = _select_vote_summary_rows(df)
    if base.empty:
        return {
            "rows_used": 0,
            "유효투표수": 0,
            "득표수": 0,
            "후보수": 0,
            "정당수": 0,
            "상위후보": None,
            "상위후보득표율": np.nan,
        }

    total_valid_votes = _sum_unique_vote_units(base)["유효투표수"].sum(min_count=1)
    candidate_count = _unique_entity_count(base, ["선거구명", "정당KEY", "정당명", "후보라벨", "후보명"])
    party_count = _unique_entity_count(base, ["정당KEY", "정당명"])
    ranking = calc_candidate_ranking(base)
    top_candidate = ranking.iloc[0] if not ranking.empty else None

    return {
        "rows_used": int(len(base)),
        "유효투표수": total_valid_votes,
        "득표수": base["득표수"].astype("Int64").sum(min_count=1),
        "후보수": candidate_count,
        "정당수": party_count,
        "상위후보": None if top_candidate is None else top_candidate["후보명"],
        "상위후보득표율": np.nan if top_candidate is None else top_candidate["득표율_계산"],
    }


def calc_votes_summary_by_election(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    base = _select_vote_summary_rows(df)
    if base.empty:
        return pd.DataFrame(columns=["선거KEY", "유효투표수", "득표수", "후보수", "정당수"] + ELECTION_META_COLUMNS[1:])

    valid_votes = _sum_unique_vote_units(base, ["선거KEY"])
    vote_sum = base.groupby("선거KEY", as_index=False, observed=True)["득표수"].sum(min_count=1).astype({"득표수": "Int64"})
    result = valid_votes.merge(vote_sum, on="선거KEY", how="left", copy=False)

    candidate_count = _grouped_unique_count(base, ["선거KEY"], ["선거구명", "정당KEY", "정당명", "후보라벨", "후보명"], "후보수")
    party_count = _grouped_unique_count(base, ["선거KEY"], ["정당KEY", "정당명"], "정당수")

    result = result.merge(candidate_count, on="선거KEY", how="left", copy=False) if not candidate_count.empty else result.assign(후보수=0)
    result = result.merge(party_count, on="선거KEY", how="left", copy=False) if not party_count.empty else result.assign(정당수=0)
    result = _attach_election_metadata(result, base, dim_election)
    result = _attach_election_label_columns(result)
    return result


def calc_candidate_ranking(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, RANKING_REQUIRED_COLUMNS, "candidate ranking frame")

    base = _select_vote_summary_rows(df)
    if base.empty:
        return pd.DataFrame(columns=["선거KEY", "정당KEY", "정당명", "후보명", "후보라벨", "득표수", "유효투표수", "득표율_계산", "순위"])

    ranking = base.groupby(["선거KEY", "정당KEY", "정당명", "후보명", "후보라벨"], dropna=False, as_index=False, observed=True)["득표수"].sum(min_count=1)
    totals = _sum_unique_vote_units(base, ["선거KEY"]).rename(columns={"유효투표수": "총유효투표수"})
    ranking = ranking.merge(totals, on="선거KEY", how="left", copy=False)
    ranking["득표율_계산"] = safe_divide(ranking["득표수"], ranking["총유효투표수"])
    ranking = ranking.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").reset_index(drop=True)
    ranking["순위"] = ranking.groupby("선거KEY", observed=True)["득표수"].rank(method="dense", ascending=False).astype("Int64")
    return ranking.rename(columns={"총유효투표수": "유효투표수"})


def _calc_group_ranking(base: pd.DataFrame, group_cols: Sequence[str], entity_type: str) -> pd.DataFrame:
    entity_cols = _entity_group_columns(entity_type)
    ranking = base.groupby([*group_cols, *entity_cols], dropna=False, as_index=False, observed=True)["득표수"].sum(min_count=1)
    totals = _sum_unique_vote_units(base, list(group_cols)).rename(columns={"유효투표수": "총유효투표수"})
    ranking = ranking.merge(totals, on=list(group_cols), how="left", copy=False)
    ranking["득표율_계산"] = safe_divide(ranking["득표수"], ranking["총유효투표수"])
    ranking = ranking.sort_values(by=[*group_cols, "득표수"], ascending=[True] * len(group_cols) + [False], kind="stable").reset_index(drop=True)
    ranking["순위"] = ranking.groupby(list(group_cols), observed=True)["득표수"].rank(method="dense", ascending=False).astype("Int64")
    return ranking


def calc_top2_gap(df: pd.DataFrame) -> pd.DataFrame:
    ranking = calc_candidate_ranking(df)
    if ranking.empty:
        return pd.DataFrame(columns=["선거KEY", "1위후보", "1위정당", "1위득표수", "1위득표율", "2위후보", "2위정당", "2위득표수", "2위득표율", "득표수격차", "득표율격차", "경쟁도지수"])

    results: list[dict[str, object]] = []
    for election_key, group in ranking.groupby("선거KEY", dropna=False, observed=True):
        top_two = group.head(2).reset_index(drop=True)
        first = top_two.iloc[0]
        second = top_two.iloc[1] if len(top_two) > 1 else None
        gap_pct = np.nan if second is None else first["득표율_계산"] - second["득표율_계산"]
        results.append(
            {
                "선거KEY": election_key,
                "1위후보": first["후보명"],
                "1위정당": first["정당명"],
                "1위득표수": first["득표수"],
                "1위득표율": first["득표율_계산"],
                "2위후보": None if second is None else second["후보명"],
                "2위정당": None if second is None else second["정당명"],
                "2위득표수": np.nan if second is None else second["득표수"],
                "2위득표율": np.nan if second is None else second["득표율_계산"],
                "득표수격차": np.nan if second is None else first["득표수"] - second["득표수"],
                "득표율격차": gap_pct,
                "경쟁도지수": np.nan if pd.isna(gap_pct) else max(0.0, 1.0 - float(gap_pct)),
            }
        )
    return pd.DataFrame(results)


def calc_entity_top2_gap(df: pd.DataFrame, entity_type: str = "후보") -> pd.DataFrame:
    base = calc_entity_share(df, entity_type=entity_type)
    if base.empty:
        return pd.DataFrame(columns=["선거KEY", "1위", "1위득표수", "1위득표율", "2위", "2위득표수", "2위득표율", "득표수격차", "득표율격차"])

    label_col = _entity_label_column(entity_type)
    results: list[dict[str, object]] = []
    for election_key, group in base.groupby("선거KEY", observed=True, dropna=False):
        group = group.sort_values(by="득표수", ascending=False, kind="stable").reset_index(drop=True)
        first = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        results.append(
            {
                "선거KEY": election_key,
                "1위": first[label_col],
                "1위득표수": first["득표수"],
                "1위득표율": first["득표율_계산"],
                "2위": None if second is None else second[label_col],
                "2위득표수": np.nan if second is None else second["득표수"],
                "2위득표율": np.nan if second is None else second["득표율_계산"],
                "득표수격차": np.nan if second is None else first["득표수"] - second["득표수"],
                "득표율격차": np.nan if second is None else first["득표율_계산"] - second["득표율_계산"],
            }
        )
    return pd.DataFrame(results)


def calc_top2_gap_trend(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    base = calc_top2_gap(df)
    if base.empty:
        return base
    base = _attach_election_metadata(base, df, dim_election)
    base = _attach_election_label_columns(base)
    return _add_period_change(base, ["득표율격차", "경쟁도지수"])


def calc_top2_gap_by_region(df: pd.DataFrame, level: str = "구시군", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    base, region_cols = _select_vote_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "선거KEY",
                *region_cols,
                "지역",
                "1위후보",
                "1위정당",
                "2위후보",
                "2위정당",
                "득표수격차",
                "득표율격차",
                "부호득표수격차",
                "부호득표율격차",
                "격차방향",
                "경쟁도지수",
            ]
        )

    group_cols = ["선거KEY", *region_cols]
    ranking = _calc_group_ranking(base, group_cols, "후보")
    results: list[dict[str, object]] = []
    for group_key, group in ranking.groupby(group_cols, dropna=False, observed=True):
        top_two = group.head(2).reset_index(drop=True)
        first = top_two.iloc[0]
        second = top_two.iloc[1] if len(top_two) > 1 else None
        row = {column: value for column, value in zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,))}
        gap_pct = np.nan if second is None else first["득표율_계산"] - second["득표율_계산"]
        gap_votes = np.nan if second is None else first["득표수"] - second["득표수"]
        gap_sign = _major_party_gap_sign(first["정당명"])
        row.update(
            {
                "1위후보": first["후보명"],
                "1위정당": first["정당명"],
                "2위후보": None if second is None else second["후보명"],
                "2위정당": None if second is None else second["정당명"],
                "득표수격차": gap_votes,
                "득표율격차": gap_pct,
                "부호득표수격차": np.nan if pd.isna(gap_votes) else float(gap_votes) * gap_sign,
                "부호득표율격차": np.nan if pd.isna(gap_pct) else float(gap_pct) * gap_sign,
                "격차방향": _gap_direction_label(first["정당명"]),
                "경쟁도지수": np.nan if pd.isna(gap_pct) else max(0.0, 1.0 - float(gap_pct)),
            }
        )
        results.append(row)

    result = pd.DataFrame(results)
    result["지역"] = build_region_label(result, region_cols)
    result = _attach_election_metadata(result, base, dim_election)
    return result.sort_values(by=["선거KEY", "득표율격차"], ascending=[True, True], kind="stable").reset_index(drop=True)


def calc_entity_pair_gap_by_region(
    df: pd.DataFrame,
    level: str = "구시군",
    entity_type: str = "후보",
    base_entity_name: str | None = None,
    compare_entity_name: str | None = None,
    dim_election: pd.DataFrame | None = None,
) -> pd.DataFrame:
    label_col = _entity_label_column(entity_type)
    region = calc_entity_share_by_region(df, entity_type=entity_type, level=level, dim_election=dim_election)
    if region.empty or not base_entity_name or not compare_entity_name:
        return pd.DataFrame()

    group_cols = ["선거KEY", *[column for column in _resolve_level_group_columns(region, level) if column in region.columns]]
    if not group_cols:
        return pd.DataFrame()

    value_cols = [*group_cols, "지역", label_col, "정당명", "득표수", "득표율_계산"]
    value_cols = [column for column in dict.fromkeys(value_cols) if column in region.columns]
    selected = region.loc[region[label_col].astype("string").isin([str(base_entity_name), str(compare_entity_name)]), value_cols].copy()
    if selected.empty:
        return pd.DataFrame()

    base = selected.loc[selected[label_col].astype("string") == str(base_entity_name)].copy()
    compare = selected.loc[selected[label_col].astype("string") == str(compare_entity_name)].copy()
    if base.empty or compare.empty:
        return pd.DataFrame()

    base_rename = {label_col: "기준대상", "득표수": "기준득표수", "득표율_계산": "기준득표율", "지역": "지역"}
    compare_rename = {label_col: "비교대상", "득표수": "비교득표수", "득표율_계산": "비교득표율", "지역": "비교지역"}
    if label_col != "정당명" and "정당명" in base.columns:
        base_rename["정당명"] = "기준정당"
        compare_rename["정당명"] = "비교정당"
    base = base.rename(columns=base_rename)
    compare = compare.rename(columns=compare_rename)
    if "기준정당" not in base.columns:
        base["기준정당"] = base["기준대상"]
    if "비교정당" not in compare.columns:
        compare["비교정당"] = compare["비교대상"]

    result = base.merge(compare, on=group_cols, how="inner", copy=False)
    if result.empty:
        return pd.DataFrame()
    if "지역" not in result.columns and "비교지역" in result.columns:
        result["지역"] = result["비교지역"]
    if "비교지역" in result.columns:
        result = result.drop(columns="비교지역")

    result["득표수격차"] = result["기준득표수"] - result["비교득표수"]
    result["득표율격차"] = result["기준득표율"] - result["비교득표율"]
    result["부호득표수격차"] = result["득표수격차"]
    result["부호득표율격차"] = result["득표율격차"]

    def _pair_gap_direction(row: pd.Series) -> str:
        gap = row["득표율격차"]
        if pd.isna(gap):
            return "비교 불가"
        if float(gap) > 0:
            return f"{row['기준대상']} 우세"
        if float(gap) < 0:
            return f"{row['비교대상']} 우세"
        return "동률"

    result["격차방향"] = result.apply(_pair_gap_direction, axis=1)
    result["경쟁도지수"] = result["득표율격차"].abs().map(lambda value: np.nan if pd.isna(value) else max(0.0, 1.0 - float(value)))
    result = _attach_election_metadata(result, region, dim_election)
    return result.sort_values(by=["선거KEY", "득표율격차"], ascending=[True, True], kind="stable").reset_index(drop=True)


def calc_entity_share(df: pd.DataFrame, entity_type: str = "후보", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    check_required_columns(df, _entity_required_columns(entity_type), "entity share frame")

    base = _select_vote_summary_rows(df)
    if base.empty:
        return pd.DataFrame(
            columns=["선거KEY", *_entity_group_columns(entity_type), "득표수", "유효투표수", "득표율_계산", *ELECTION_META_COLUMNS[1:], "선거라벨", "선거축라벨"]
        )
    base = _normalize_entity_labels(base, entity_type)

    entity_cols = _entity_group_columns(entity_type)
    grouped = base.groupby(["선거KEY", *entity_cols], dropna=False, as_index=False, observed=True)["득표수"].sum(min_count=1)
    totals = _sum_unique_vote_units(base, ["선거KEY"])
    grouped = grouped.merge(totals, on="선거KEY", how="left", copy=False)
    grouped["득표율_계산"] = safe_divide(grouped["득표수"], grouped["유효투표수"])
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return grouped.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").reset_index(drop=True)


def calc_party_share(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    return calc_entity_share(df, entity_type="정당", dim_election=dim_election)


def calc_entity_share_by_region(
    df: pd.DataFrame,
    entity_type: str = "후보",
    level: str = "읍면동",
    dim_election: pd.DataFrame | None = None,
) -> pd.DataFrame:
    check_required_columns(df, _entity_required_columns(entity_type), "entity share by region frame")

    base, region_cols = _select_vote_rows_by_level(df, level)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "선거KEY",
                *region_cols,
                *_entity_group_columns(entity_type),
                "득표수",
                "유효투표수",
                "득표율_계산",
                "지역",
                *ELECTION_META_COLUMNS[1:],
                "선거라벨",
                "선거축라벨",
            ]
        )
    base = _normalize_entity_labels(base, entity_type)

    entity_cols = _entity_group_columns(entity_type)
    grouped = base.groupby(["선거KEY", *region_cols, *entity_cols], dropna=False, as_index=False, observed=True)["득표수"].sum(min_count=1)
    totals = _sum_unique_vote_units(base, ["선거KEY", *region_cols])
    grouped = grouped.merge(totals, on=["선거KEY", *region_cols], how="left", copy=False)
    grouped["득표율_계산"] = safe_divide(grouped["득표수"], grouped["유효투표수"])
    grouped["지역"] = build_region_label(grouped, region_cols)
    grouped = _attach_election_metadata(grouped, base, dim_election)
    grouped = _attach_election_label_columns(grouped)
    return grouped.sort_values(by=["선거KEY", "지역", "득표수"], ascending=[True, True, False], kind="stable").reset_index(drop=True)


def calc_candidate_share_by_region(df: pd.DataFrame, level: str = "읍면동", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    return calc_entity_share_by_region(df, entity_type="후보", level=level, dim_election=dim_election)


def calc_entity_trend(df: pd.DataFrame, entity_type: str = "후보", dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    result = calc_entity_share(df, entity_type=entity_type, dim_election=dim_election)
    label_column = _entity_label_column(entity_type)
    return _add_period_change(result, ["득표율_계산", "득표수"], [label_column])


def calc_entity_trend_by_region(
    df: pd.DataFrame,
    entity_type: str = "후보",
    level: str = "구시군",
    dim_election: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = calc_entity_share_by_region(df, entity_type=entity_type, level=level, dim_election=dim_election)
    label_column = _entity_label_column(entity_type)
    return _add_period_change(result, ["득표율_계산", "득표수"], ["지역", label_column])


def _aggregate_vote_rowtype_components(
    df: pd.DataFrame,
    group_cols: Sequence[str] | None = None,
    entity_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    check_required_columns(df, ["RowType", "득표수", "유효투표수"] + VOTE_UNIT_COLUMNS, "vote rowtype component frame")

    requested_group_cols = [column for column in (group_cols or []) if column in df.columns]
    requested_entity_cols = [column for column in (entity_cols or []) if column in df.columns]
    merge_group_cols = [f"__merge_group_{idx}" for idx, _ in enumerate(requested_group_cols)] or ["__all__"]
    merge_entity_cols = [f"__merge_entity_{idx}" for idx, _ in enumerate(requested_entity_cols)]

    def _with_merge_keys(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if requested_group_cols:
            for merge_col, source_col in zip(merge_group_cols, requested_group_cols):
                prepared[merge_col] = prepared[source_col].astype("string").fillna(f"__missing_group__{source_col}")
        else:
            prepared["__all__"] = 1
        for merge_col, source_col in zip(merge_entity_cols, requested_entity_cols):
            prepared[merge_col] = prepared[source_col].astype("string").fillna(f"__missing_entity__{source_col}")
        return prepared

    local_total = _with_merge_keys(df.loc[df["RowType"].eq("읍면동")].copy())
    local_early = _with_merge_keys(df.loc[df["RowType"].eq("관내사전투표")].copy())
    outside = _with_merge_keys(df.loc[df["RowType"].eq("관외사전투표")].copy())
    direct_day = _with_merge_keys(_select_direct_day_rows(df))

    key_frames = [
        frame.loc[:, [*merge_group_cols, *merge_entity_cols, *requested_group_cols, *requested_entity_cols]]
        for frame in [local_total, local_early, outside, direct_day]
        if not frame.empty
    ]
    if not key_frames:
        return pd.DataFrame(columns=[*requested_group_cols, *requested_entity_cols, "RowType", "득표수", "유효투표수"])

    key_frame = (
        pd.concat(key_frames, ignore_index=True)
        .drop_duplicates(subset=[*merge_group_cols, *merge_entity_cols])
        .reset_index(drop=True)
    )

    def _group_votes(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[*merge_group_cols, *merge_entity_cols, f"{prefix}_득표수"])
        grouped = frame.groupby([*merge_group_cols, *merge_entity_cols], as_index=False, observed=True)["득표수"].sum(min_count=1)
        return grouped.rename(columns={"득표수": f"{prefix}_득표수"})

    def _group_valid(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[*merge_group_cols, f"{prefix}_유효투표수"])
        drop_cols = [column for column in [*merge_group_cols, *merge_entity_cols] if column in frame.columns]
        source = frame.drop(columns=drop_cols)
        grouped = _sum_unique_vote_units(source, requested_group_cols or None)
        if requested_group_cols:
            for merge_col, source_col in zip(merge_group_cols, requested_group_cols):
                grouped[merge_col] = grouped[source_col].astype("string").fillna(f"__missing_group__{source_col}")
        else:
            grouped["__all__"] = 1
        return grouped.rename(columns={"유효투표수": f"{prefix}_유효투표수"})

    component_base = key_frame.copy()
    component_base = component_base.merge(_group_votes(local_total, "local_total"), on=[*merge_group_cols, *merge_entity_cols], how="left", copy=False)
    component_base = component_base.merge(_group_votes(local_early, "local_early"), on=[*merge_group_cols, *merge_entity_cols], how="left", copy=False)
    component_base = component_base.merge(_group_votes(outside, "outside"), on=[*merge_group_cols, *merge_entity_cols], how="left", copy=False)
    component_base = component_base.merge(_group_votes(direct_day, "direct_day"), on=[*merge_group_cols, *merge_entity_cols], how="left", copy=False)
    component_base = component_base.merge(_group_valid(local_total, "local_total"), on=merge_group_cols, how="left", copy=False)
    component_base = component_base.merge(_group_valid(local_early, "local_early"), on=merge_group_cols, how="left", copy=False)
    component_base = component_base.merge(_group_valid(outside, "outside"), on=merge_group_cols, how="left", copy=False)
    component_base = component_base.merge(_group_valid(direct_day, "direct_day"), on=merge_group_cols, how="left", copy=False)

    for prefix, metric in [
        ("local_total", "득표수"),
        ("local_early", "득표수"),
        ("outside", "득표수"),
        ("direct_day", "득표수"),
        ("local_total", "유효투표수"),
        ("local_early", "유효투표수"),
        ("outside", "유효투표수"),
        ("direct_day", "유효투표수"),
    ]:
        column_name = f"{prefix}_{metric}"
        if column_name not in component_base.columns:
            component_base[column_name] = pd.NA

    local_total_available = component_base["local_total_유효투표수"].notna() | component_base["local_total_득표수"].notna()
    direct_day_available = component_base["direct_day_유효투표수"].notna() | component_base["direct_day_득표수"].notna()

    component_base["관내사전투표_득표수"] = component_base["local_early_득표수"].fillna(0)
    component_base["관외사전투표_득표수"] = component_base["outside_득표수"].fillna(0)
    component_base["관내사전투표_유효투표수"] = component_base["local_early_유효투표수"].fillna(0)
    component_base["관외사전투표_유효투표수"] = component_base["outside_유효투표수"].fillna(0)

    # 선거일 개표 행이 있으면 그 값을 우선 사용한다.
    # 읍면동-관내사전투표 차감은 보조 fallback으로만 써야 미세한 %p 오차가 남지 않는다.
    derived_day_votes = (component_base["local_total_득표수"].fillna(0) - component_base["local_early_득표수"].fillna(0)).clip(lower=0)
    derived_day_valid = (component_base["local_total_유효투표수"].fillna(0) - component_base["local_early_유효투표수"].fillna(0)).clip(lower=0)
    election_day_votes = component_base["direct_day_득표수"].fillna(0).where(direct_day_available, derived_day_votes.where(local_total_available, 0))
    election_day_valid = component_base["direct_day_유효투표수"].fillna(0).where(direct_day_available, derived_day_valid.where(local_total_available, 0))
    component_base["선거일투표_득표수"] = election_day_votes
    component_base["선거일투표_유효투표수"] = election_day_valid

    result_parts: list[pd.DataFrame] = []
    for rowtype in TURNOUT_COMPONENT_ORDER:
        part = component_base.loc[:, [*requested_group_cols, *requested_entity_cols]].copy()
        part["RowType"] = rowtype
        part["득표수"] = component_base[f"{rowtype}_득표수"]
        part["유효투표수"] = component_base[f"{rowtype}_유효투표수"]
        result_parts.append(part)

    result = pd.concat(result_parts, ignore_index=True)
    return _sort_turnout_components(result)


def calc_votes_by_rowtype(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _aggregate_vote_rowtype_components(df)
    if normalized.empty:
        return pd.DataFrame(columns=["RowType", "득표수", "유효투표수", "득표율_계산", "득표구성비"])
    normalized["득표율_계산"] = safe_divide(normalized["득표수"], normalized["유효투표수"])
    normalized["득표구성비"] = safe_divide(normalized["득표수"], normalized["득표수"].sum(min_count=1))
    return _sort_turnout_components(normalized)


def calc_rowtype_vote_trend(df: pd.DataFrame, dim_election: pd.DataFrame | None = None) -> pd.DataFrame:
    normalized = _aggregate_vote_rowtype_components(df, ["선거KEY"])
    if normalized.empty:
        return pd.DataFrame(columns=["선거KEY", "RowType", "득표수", "유효투표수", "득표율_계산", "득표구성비"] + ELECTION_META_COLUMNS[1:])
    normalized["득표율_계산"] = safe_divide(normalized["득표수"], normalized["유효투표수"])
    election_totals = normalized.groupby("선거KEY", as_index=False, observed=True)["득표수"].sum(min_count=1).rename(columns={"득표수": "총득표수"})
    normalized = normalized.merge(election_totals, on="선거KEY", how="left", copy=False)
    normalized["득표구성비"] = safe_divide(normalized["득표수"], normalized["총득표수"])
    normalized = normalized.drop(columns="총득표수")
    normalized = _attach_election_metadata(normalized, df, dim_election)
    normalized = _attach_election_label_columns(normalized)
    normalized = _add_period_change(normalized, ["득표율_계산", "득표구성비"], ["RowType"])
    return _sort_turnout_components(normalized)


def calc_rowtype_entity_breakdown(df: pd.DataFrame, entity_type: str = "후보") -> pd.DataFrame:
    check_required_columns(df, _entity_required_columns(entity_type), "rowtype entity breakdown frame")

    entity_cols = _entity_group_columns(entity_type)
    normalized_df = _normalize_entity_labels(df, entity_type)
    grouped = _aggregate_vote_rowtype_components(normalized_df, entity_cols=entity_cols)
    if grouped.empty:
        return pd.DataFrame(columns=["RowType", *entity_cols, "득표수", "유효투표수", "득표율_계산", "득표구성비"])
    grouped["득표율_계산"] = safe_divide(grouped["득표수"], grouped["유효투표수"])
    grouped["득표구성비"] = safe_divide(grouped["득표수"], grouped.groupby("RowType", observed=True)["득표수"].transform("sum"))

    summary_base = _select_vote_summary_rows(normalized_df)
    if not summary_base.empty:
        total_valid_votes = _sum_unique_vote_units(summary_base)["유효투표수"].sum(min_count=1)
        total_grouped = summary_base.groupby(entity_cols, dropna=False, as_index=False, observed=True)["득표수"].sum(min_count=1)
        total_grouped["RowType"] = "전체 합계"
        total_grouped["유효투표수"] = total_valid_votes
        total_grouped["득표율_계산"] = safe_divide(total_grouped["득표수"], total_grouped["유효투표수"])
        total_grouped["득표구성비"] = safe_divide(total_grouped["득표수"], total_grouped["득표수"].sum(min_count=1))
        grouped = pd.concat(
            [
                total_grouped.loc[:, ["RowType", *entity_cols, "득표수", "유효투표수", "득표율_계산", "득표구성비"]],
                grouped.loc[:, ["RowType", *entity_cols, "득표수", "유효투표수", "득표율_계산", "득표구성비"]],
            ],
            ignore_index=True,
        )
    return _sort_turnout_components(grouped.sort_values(by=["RowType", "득표수"], ascending=[True, False], kind="stable").reset_index(drop=True))


def calc_distribution_by_region(df: pd.DataFrame, entity_type: str = "후보", level: str = "읍면동") -> pd.DataFrame:
    return calc_entity_share_by_region(df, entity_type=entity_type, level=level)


def calc_turnout_vote_scatter(
    turnout_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    level: str = "구시군",
    entity_type: str = "후보",
    entity_name: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    turnout_region = calc_turnout_by_region(turnout_df, level=level)
    votes_region = calc_entity_share_by_region(votes_df, entity_type=entity_type, level=level)
    label_column = _entity_label_column(entity_type)
    if entity_name:
        if isinstance(entity_name, Sequence) and not isinstance(entity_name, str):
            selected_names = [str(value) for value in entity_name if value not in (None, "")]
            votes_region = votes_region.loc[votes_region[label_column].astype("string").isin(selected_names)].copy()
        else:
            votes_region = votes_region.loc[votes_region[label_column].astype("string") == str(entity_name)].copy()
    else:
        votes_region = votes_region.sort_values(by=["지역", "득표수"], ascending=[True, False], kind="stable").drop_duplicates(subset=["지역"])

    join_cols = [column for column in _resolve_level_group_columns(turnout_region, level) if column in votes_region.columns]
    if not join_cols:
        empty_columns = ["지역", "투표율", "득표율_계산", "투표수", "득표수", label_column]
        if "정당명" in votes_region.columns and "정당명" not in empty_columns:
            empty_columns.append("정당명")
        return pd.DataFrame(columns=empty_columns)

    vote_cols = [*join_cols, label_column, "득표수", "득표율_계산", "지역"]
    if "정당명" in votes_region.columns and "정당명" not in vote_cols:
        vote_cols.append("정당명")

    merged = turnout_region.merge(
        votes_region.loc[:, vote_cols],
        on=join_cols,
        how="inner",
        copy=False,
    )
    if "지역" not in merged.columns:
        if "지역_y" in merged.columns:
            merged["지역"] = merged["지역_y"]
        elif "지역_x" in merged.columns:
            merged["지역"] = merged["지역_x"]
    return merged.sort_values(by="투표율", ascending=False, kind="stable").reset_index(drop=True)


def calc_polling_station_metrics(turnout_df: pd.DataFrame, votes_df: pd.DataFrame, polling_df: pd.DataFrame) -> pd.DataFrame:
    turnout_base = turnout_df.loc[turnout_df["투표소KEY"].notna()].copy()
    votes_base = votes_df.loc[votes_df["투표소KEY"].notna()].copy()
    if turnout_base.empty and votes_base.empty:
        return polling_df.iloc[0:0].copy()

    turnout_grouped = turnout_base.groupby("투표소KEY", as_index=False, observed=True)[TURNOUT_NUMERIC_COLUMNS].sum(min_count=1)
    turnout_grouped = _add_turnout_rates(turnout_grouped)
    turnout_mix_source = turnout_df.loc[turnout_df["읍면동KEY"].notna()].copy() if "읍면동KEY" in turnout_df.columns else turnout_df.iloc[0:0].copy()
    turnout_mix = _aggregate_turnout_components(turnout_mix_source, ["읍면동KEY"])
    if turnout_mix.empty:
        turnout_mix_wide = pd.DataFrame(columns=["읍면동KEY", "사전투표 비중", "선거일투표 비중"])
    else:
        turnout_mix_wide = turnout_mix.pivot_table(index="읍면동KEY", columns="RowType", values="투표구성비", aggfunc="first").reset_index()
        for rowtype in ["관내사전투표", "관외사전투표", "선거일투표"]:
            if rowtype not in turnout_mix_wide.columns:
                turnout_mix_wide[rowtype] = 0.0
        turnout_mix_wide["사전투표 비중"] = turnout_mix_wide["관내사전투표"].fillna(0) + turnout_mix_wide["관외사전투표"].fillna(0)
        turnout_mix_wide["선거일투표 비중"] = turnout_mix_wide["선거일투표"].fillna(0)
        turnout_mix_wide = turnout_mix_wide.loc[:, ["읍면동KEY", "사전투표 비중", "선거일투표 비중"]]

    vote_rank = pd.DataFrame(
        columns=[
            "투표소KEY",
            "1위후보",
            "1위정당",
            "1위득표수",
            "1위득표율",
            "2위후보",
            "2위정당",
            "2위득표수",
            "2위득표율",
            "득표수격차",
            "득표율격차",
            "경쟁도지수",
        ]
    )
    if not votes_base.empty:
        ranking = _calc_group_ranking(votes_base, ["투표소KEY"], "후보")
        rows: list[dict[str, object]] = []
        for polling_key, group in ranking.groupby("투표소KEY", observed=True, dropna=False):
            top_two = group.head(2).reset_index(drop=True)
            first = top_two.iloc[0]
            second = top_two.iloc[1] if len(top_two) > 1 else None
            gap_pct = np.nan if second is None else first["득표율_계산"] - second["득표율_계산"]
            gap_votes = np.nan if second is None else first["득표수"] - second["득표수"]
            rows.append(
                {
                    "투표소KEY": polling_key,
                    "1위후보": first["후보명"],
                    "1위정당": first["정당명"],
                    "1위득표수": first["득표수"],
                    "1위득표율": first["득표율_계산"],
                    "2위후보": None if second is None else second["후보명"],
                    "2위정당": None if second is None else second["정당명"],
                    "2위득표수": np.nan if second is None else second["득표수"],
                    "2위득표율": np.nan if second is None else second["득표율_계산"],
                    "득표수격차": gap_votes,
                    "득표율격차": gap_pct,
                    "경쟁도지수": np.nan if pd.isna(gap_pct) else max(0.0, 1.0 - float(gap_pct)),
                }
            )
        vote_rank = pd.DataFrame(rows)

    result = polling_df.merge(turnout_grouped, on="투표소KEY", how="left", copy=False)
    mix_join_col = "읍면동KEY_D" if "읍면동KEY_D" in result.columns else "읍면동KEY"
    if mix_join_col in result.columns:
        result = result.merge(turnout_mix_wide, left_on=mix_join_col, right_on="읍면동KEY", how="left", copy=False)
        if "읍면동KEY" in result.columns and mix_join_col != "읍면동KEY":
            result = result.drop(columns="읍면동KEY")
    else:
        result["사전투표 비중"] = np.nan
        result["선거일투표 비중"] = np.nan
    result = result.merge(vote_rank, on="투표소KEY", how="left", copy=False)
    return result.sort_values(by=["투표수", "1위득표수"], ascending=[False, False], kind="stable").reset_index(drop=True)


def calc_map_metric_by_region(
    turnout_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    level: str,
    metric_name: str,
    entity_type: str | None = None,
    entity_name: str | None = None,
    compare_entity_name: str | None = None,
) -> pd.DataFrame:
    normalized_level = _normalize_level(level)
    if metric_name in {"투표율", "유효투표율", "무효투표율", "기권율"}:
        region = calc_turnout_by_region(turnout_df, level=normalized_level)
        region["지도지표"] = region[metric_name]
        return region

    if metric_name in {"경쟁도지수", "득표수격차", "득표율격차"}:
        region = calc_top2_gap_by_region(votes_df, level=normalized_level)
        if metric_name == "득표수격차" and "부호득표수격차" in region.columns:
            region["지도지표"] = region["부호득표수격차"]
        elif metric_name == "득표율격차" and "부호득표율격차" in region.columns:
            region["지도지표"] = region["부호득표율격차"]
        else:
            region["지도지표"] = region[metric_name]
        return region

    if metric_name in {"후보 간 득표수 격차", "후보 간 득표율 격차", "정당 간 득표수 격차", "정당 간 득표율 격차"}:
        pair_entity_type = "정당" if metric_name.startswith("정당") else "후보"
        region = calc_entity_pair_gap_by_region(
            votes_df,
            level=normalized_level,
            entity_type=pair_entity_type,
            base_entity_name=entity_name,
            compare_entity_name=compare_entity_name,
        )
        if region.empty:
            return region
        region["지도지표"] = region["득표수격차"] if "득표수" in metric_name else region["득표율격차"]
        return region

    if metric_name == "사전투표 비중":
        mix = calc_turnout_rowtype_mix_by_region(turnout_df, level=normalized_level)
        if mix.empty:
            return mix
        early = mix.loc[mix["RowType"].isin(["관내사전투표", "관외사전투표"])].copy()
        region_cols = [column for column in _resolve_level_group_columns(early, normalized_level) if column in early.columns]
        grouped = early.groupby(region_cols + ["지역"], as_index=False, observed=True)["투표구성비"].sum(min_count=1)
        grouped["지도지표"] = grouped["투표구성비"]
        return grouped

    if metric_name in {"정당 득표수", "후보 득표수", "정당 득표율", "후보 득표율"}:
        if entity_type is None:
            entity_type = "정당" if metric_name.startswith("정당") else "후보"
        region = calc_entity_share_by_region(votes_df, entity_type=entity_type, level=normalized_level)
        label_column = _entity_label_column(entity_type)
        if entity_name:
            region = region.loc[region[label_column].astype("string") == str(entity_name)].copy()
        region["지도지표"] = region["득표수"] if metric_name.endswith("득표수") else region["득표율_계산"]
        return region

    return pd.DataFrame()
