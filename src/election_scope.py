from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


SUMMARY_VALUE = "합계"


@dataclass(frozen=True)
class ElectionScope:
    key: str
    label: str
    level: str
    filters: Mapping[str, str]


def local_vote_scope_level(election_key: str | None, election_kind: str | None) -> str | None:
    """Return the official vote-counting race scope for multi-race local election keys."""
    kind = str(election_kind or "")
    key_suffix = str(election_key or "").split("-")[-1]

    if "광역단체장" in kind or "광역의원비례대표" in kind or key_suffix in {"L1", "L3"}:
        return "시도"
    if "기초단체장" in kind or "기초의원비례대표" in kind or key_suffix in {"L2", "L4"}:
        return "구시군"
    if "광역의회" in kind or "기초의회" in kind or key_suffix in {"L5", "L6"}:
        return "선거구"
    return None


def _first_text(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None
    values = df[column].dropna().astype("string")
    values = values.loc[values.ne("") & values.ne(SUMMARY_VALUE)]
    return None if values.empty else str(values.iloc[0])


def _is_concrete(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.notna() & values.ne("") & values.ne(SUMMARY_VALUE)


def _scope_key(filters: Mapping[str, str]) -> str:
    return "|".join(f"{column}={value}" for column, value in sorted(filters.items()))


def _label_from_values(values: list[str]) -> str:
    return " ".join(value for value in values if value)


def _records_to_scopes(records: pd.DataFrame, level: str, label_columns: list[str], filter_columns: list[str]) -> list[ElectionScope]:
    scopes: list[ElectionScope] = []
    for record in records.to_dict("records"):
        filters = {
            column: str(record[column])
            for column in filter_columns
            if column in record and pd.notna(record[column]) and str(record[column]) not in ("", SUMMARY_VALUE)
        }
        if not filters:
            continue
        label = _label_from_values(
            [
                str(record[column])
                for column in label_columns
                if column in record and pd.notna(record[column]) and str(record[column]) not in ("", SUMMARY_VALUE)
            ]
        )
        if not label:
            label = _label_from_values(list(filters.values()))
        scopes.append(ElectionScope(key=_scope_key(filters), label=label, level=level, filters=filters))
    return scopes


def build_local_vote_scope_options(df: pd.DataFrame) -> list[ElectionScope]:
    if df.empty or "선거KEY" not in df.columns:
        return []

    election_key = _first_text(df, "선거KEY")
    election_kind = _first_text(df, "선거종류")
    level = local_vote_scope_level(election_key, election_kind)
    if level is None:
        return []

    if level == "시도":
        if "시도명" not in df.columns:
            return []
        records = df.loc[_is_concrete(df["시도명"]), ["시도명"]].drop_duplicates(ignore_index=True)
        return _records_to_scopes(records, level, ["시도명"], ["시도명"])

    if level == "구시군":
        required = [column for column in ["시도명", "구시군명", "구시군KEY"] if column in df.columns]
        if "구시군명" not in required:
            return []
        mask = _is_concrete(df["구시군명"])
        if "시도명" in df.columns:
            mask &= _is_concrete(df["시도명"])
        records = df.loc[mask, required].drop_duplicates(ignore_index=True)
        filter_columns = ["시도명", "구시군명"]
        if "구시군KEY" in records.columns and records["구시군KEY"].notna().any():
            filter_columns.append("구시군KEY")
        return _records_to_scopes(records, level, ["시도명", "구시군명"], filter_columns)

    if level == "선거구":
        if "선거구명" not in df.columns:
            return []
        required = [column for column in ["시도명", "구시군명", "구시군KEY", "선거구명"] if column in df.columns]
        mask = _is_concrete(df["선거구명"])
        if "시도명" in df.columns:
            mask &= _is_concrete(df["시도명"])
        records = df.loc[mask, required].drop_duplicates(ignore_index=True)
        filter_columns = [column for column in ["시도명", "구시군명", "선거구명"] if column in records.columns]
        return _records_to_scopes(records, level, ["시도명", "구시군명", "선거구명"], filter_columns)

    return []


def preferred_scope_key(scopes: list[ElectionScope], selected: Mapping[str, list[str]]) -> str | None:
    if not scopes:
        return None

    selected_sido = {str(value) for value in selected.get("시도명", [])}
    selected_gusigun = {str(value) for value in selected.get("구시군명", [])}
    selected_dong = {str(value) for value in selected.get("읍면동명", [])}

    def matches(scope: ElectionScope) -> bool:
        if selected_sido and scope.filters.get("시도명") not in selected_sido:
            return False
        if selected_gusigun and scope.filters.get("구시군명") not in selected_gusigun:
            return False
        if selected_dong and scope.level not in {"구시군", "선거구"}:
            return False
        return True

    for scope in scopes:
        if matches(scope):
            return scope.key
    return scopes[0].key


def filter_by_election_scope(df: pd.DataFrame, scope: ElectionScope | None) -> pd.DataFrame:
    if scope is None or df.empty:
        return df

    result = df
    for column, value in scope.filters.items():
        if column in result.columns:
            result = result.loc[result[column].astype("string").eq(str(value))]
    return result.copy()
