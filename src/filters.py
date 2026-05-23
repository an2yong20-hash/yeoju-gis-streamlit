from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import streamlit as st

FILTER_COLUMNS = ["선거KEY", "선거시점", "선거명", "선거종류", "시도명", "구시군명", "일반구명", "읍면동명", "RowType"]
PARTY_FILTER_COLUMNS = FILTER_COLUMNS + ["정당명"]
CANDIDATE_FILTER_COLUMNS = PARTY_FILTER_COLUMNS + ["후보명"]
HIERARCHY_FILTER_COLUMNS = ["시도명", "구시군명", "일반구명", "읍면동명", "RowType"]
ENTITY_FILTER_COLUMNS = ["정당명", "후보명"]

GLOBAL_FILTER_DIM_COLUMNS = {
    "DimElection": ["선거KEY", "선거시점", "선거명", "선거종류"],
}

GLOBAL_FILTER_FACT_COLUMNS = {
    "FactTurnout": ["선거KEY", "시도명", "구시군명", "구시군KEY", "일반구명", "읍면동명", "RowType"],
    "FactVotes": ["선거KEY", "시도명", "구시군명", "구시군KEY", "일반구명", "읍면동명", "RowType", "정당명", "후보명"],
}

FILTER_ALIASES = {
    "선거KEY": ["선거KEY"],
    "선거시점": ["선거시점"],
    "선거명": ["선거명"],
    "선거종류": ["선거종류"],
    "시도명": ["시도명", "시도", "시도명_D", "시도명_F"],
    "구시군명": ["구시군명", "구시군", "구시군명_D", "구시군명_F"],
    "일반구명": ["일반구명", "일반구명_D", "일반구명_F"],
    "읍면동명": ["읍면동명", "읍면동명_D", "읍면동명_F"],
    "RowType": ["RowType"],
    "정당명": ["정당명"],
    "후보명": ["후보명"],
}

FILTER_WIDGET_KEYS = {
    "선거시점": "global_filter_years",
    "선거라벨": "global_filter_elections",
    "시도명": "global_filter_sido",
    "구시군명": "global_filter_gusigun",
    "인접구시군명": "global_filter_adjacent_gusigun",
    "일반구명": "global_filter_gu",
    "읍면동명": "global_filter_dong",
    "RowType": "global_filter_rowtype",
    "정당명": "global_filter_party",
    "후보명": "global_filter_candidate",
}
ADJACENT_SOURCE_STATE_KEY = "global_filter_adjacent_source_key"


def _categorize_filter_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df
    for column in columns:
        if column in result.columns and not isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].astype("category")
    return result


def _prepare_filter_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    ordered_columns = list(columns)
    available_columns = [column for column in ordered_columns if column in df.columns]
    result = df.loc[:, available_columns].copy()
    for column in ordered_columns:
        if column not in result.columns:
            result[column] = pd.NA
    result = result.loc[:, ordered_columns].dropna(how="all").drop_duplicates().reset_index(drop=True)
    return _categorize_filter_columns(result, ordered_columns)


def _sorted_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype("string").unique().tolist())


def _ensure_filter_state() -> None:
    for widget_key in FILTER_WIDGET_KEYS.values():
        if widget_key not in st.session_state or not isinstance(st.session_state[widget_key], list):
            st.session_state[widget_key] = []


def _sanitize_widget_state(widget_key: str, options: list[str]) -> list[str]:
    current_values = [str(value) for value in st.session_state.get(widget_key, [])]
    valid_values = {str(value) for value in options}
    sanitized = [value for value in current_values if value in valid_values]
    st.session_state[widget_key] = sanitized
    return sanitized


def _current_selection(column: str) -> list[str]:
    return [str(value) for value in st.session_state.get(FILTER_WIDGET_KEYS[column], [])]


def _filter_by_selected(frame: pd.DataFrame, selected: dict[str, list[str]]) -> pd.DataFrame:
    result = frame

    if selected.get("선거시점") and "선거시점" in result.columns:
        result = result.loc[result["선거시점"].astype("string").isin(selected["선거시점"])]
    if selected.get("선거KEY") and "선거KEY" in result.columns:
        result = result.loc[result["선거KEY"].astype("string").isin(selected["선거KEY"])]

    for column in HIERARCHY_FILTER_COLUMNS + ENTITY_FILTER_COLUMNS:
        if selected.get(column) and column in result.columns:
            result = result.loc[result[column].astype("string").isin(selected[column])]

    return result


def _build_gusigun_lookup(facts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lookup_frames: list[pd.DataFrame] = []
    for fact_df in facts.values():
        required_columns = [column for column in ["시도명", "구시군명", "구시군KEY"] if column in fact_df.columns]
        if len(required_columns) != 3:
            continue
        lookup_frames.append(fact_df.loc[:, required_columns].copy())

    if not lookup_frames:
        return pd.DataFrame(columns=["시도명", "구시군명", "구시군KEY"])

    lookup = pd.concat(lookup_frames, ignore_index=True)
    lookup = lookup.dropna(subset=["시도명", "구시군명", "구시군KEY"]).copy()
    lookup["시도명"] = lookup["시도명"].astype("string")
    lookup["구시군명"] = lookup["구시군명"].astype("string")
    lookup["구시군KEY"] = lookup["구시군KEY"].astype("string")
    lookup = lookup.loc[
        ~lookup["구시군KEY"].str.contains("합계", na=False)
        & ~lookup["구시군명"].str.contains("합계", na=False)
    ].drop_duplicates()
    return lookup.sort_values(by=["시도명", "구시군명"], kind="stable").reset_index(drop=True)


def build_filter_options(facts: dict[str, pd.DataFrame], dims: dict[str, pd.DataFrame]) -> dict[str, object]:
    election_df = dims["DimElection"].loc[:, ["선거KEY", "선거시점", "선거명", "선거종류"]].drop_duplicates().copy()
    election_df["선거KEY"] = election_df["선거KEY"].astype("string")
    election_df["선거시점"] = election_df["선거시점"].astype("string")
    election_df["선거명"] = election_df["선거명"].astype("string")
    election_df["선거종류"] = election_df["선거종류"].astype("string")
    election_df["label"] = election_df["선거KEY"].str.cat(election_df["선거명"], sep=" | ")
    election_df = election_df.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False], kind="stable").reset_index(drop=True)
    election_df = _categorize_filter_columns(election_df, ["선거KEY", "선거시점", "선거명", "선거종류", "label"])
    election_meta = election_df.loc[:, ["선거KEY", "선거시점", "선거명", "선거종류", "label"]].drop_duplicates(subset=["선거KEY"])

    common_frames: list[pd.DataFrame] = []
    for fact_df in facts.values():
        common_frame = _prepare_filter_frame(
            fact_df,
            ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "RowType"],
        )
        common_frame = common_frame.merge(election_meta, on="선거KEY", how="left", copy=False)
        common_frames.append(_prepare_filter_frame(common_frame, FILTER_COLUMNS + ["label"]))

    common_base = (
        _categorize_filter_columns(pd.concat(common_frames, ignore_index=True).drop_duplicates().reset_index(drop=True), FILTER_COLUMNS + ["label"])
        if common_frames
        else pd.DataFrame(columns=FILTER_COLUMNS + ["label"])
    )

    if "FactVotes" in facts:
        party_base = _prepare_filter_frame(
            facts["FactVotes"],
            ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "RowType", "정당명"],
        )
        party_base = party_base.merge(election_meta, on="선거KEY", how="left", copy=False)
        party_base = _prepare_filter_frame(party_base, PARTY_FILTER_COLUMNS + ["label"])

        candidate_base = _prepare_filter_frame(
            facts["FactVotes"],
            ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "RowType", "정당명", "후보명"],
        )
        candidate_base = candidate_base.merge(election_meta, on="선거KEY", how="left", copy=False)
        candidate_base = _prepare_filter_frame(candidate_base, CANDIDATE_FILTER_COLUMNS + ["label"])
    else:
        party_base = pd.DataFrame(columns=PARTY_FILTER_COLUMNS + ["label"])
        candidate_base = pd.DataFrame(columns=CANDIDATE_FILTER_COLUMNS + ["label"])

    gusigun_lookup = _build_gusigun_lookup(facts)

    return {
        "elections": election_df,
        "common_base": common_base,
        "party_base": _categorize_filter_columns(party_base, PARTY_FILTER_COLUMNS + ["label"]),
        "candidate_base": _categorize_filter_columns(candidate_base, CANDIDATE_FILTER_COLUMNS + ["label"]),
        "gusigun_lookup": gusigun_lookup,
        "election_label_to_key": dict(zip(election_df["label"].astype("string"), election_df["선거KEY"].astype("string"))),
        "election_key_to_name": dict(zip(election_df["선거KEY"].astype("string"), election_df["선거명"].astype("string"))),
        "election_key_to_label": dict(zip(election_df["선거KEY"].astype("string"), election_df["label"].astype("string"))),
    }


def render_sidebar_filters(
    filter_options: dict[str, object],
    include_party: bool = True,
    include_candidate: bool = True,
    show_election_filters: bool = True,
    title: str = "공통 필터",
) -> dict[str, list[str]]:
    _ensure_filter_state()

    sidebar = st.sidebar
    sidebar.subheader(title)

    elections = filter_options["elections"]
    common_base = filter_options["common_base"]
    party_base = filter_options["party_base"]
    candidate_base = filter_options["candidate_base"]
    gusigun_lookup = filter_options.get("gusigun_lookup", pd.DataFrame(columns=["시도명", "구시군명", "구시군KEY"]))
    election_label_to_key = filter_options["election_label_to_key"]

    selected: dict[str, list[str]] = {
        "선거시점": [],
        "선거KEY": [],
        "선거명": [],
        "선거종류": [],
        "시도명": [],
        "구시군명": [],
        "인접구시군명": _current_selection("인접구시군명"),
        "일반구명": [],
        "읍면동명": [],
        "RowType": [],
        "정당명": _current_selection("정당명"),
        "후보명": _current_selection("후보명"),
    }

    election_candidates = elections
    if show_election_filters:
        year_options = _sorted_options(elections, "선거시점")
        _sanitize_widget_state(FILTER_WIDGET_KEYS["선거시점"], year_options)
        sidebar.multiselect("선거시점", options=year_options, key=FILTER_WIDGET_KEYS["선거시점"])
        selected["선거시점"] = _current_selection("선거시점")

        if selected["선거시점"]:
            election_candidates = election_candidates.loc[
                election_candidates["선거시점"].astype("string").isin(selected["선거시점"])
            ]

        election_label_options = election_candidates["label"].astype("string").tolist()
        _sanitize_widget_state(FILTER_WIDGET_KEYS["선거라벨"], election_label_options)
        sidebar.multiselect("선거 선택", options=election_label_options, key=FILTER_WIDGET_KEYS["선거라벨"])

        selected_labels = [str(value) for value in st.session_state[FILTER_WIDGET_KEYS["선거라벨"]]]
        selected["선거KEY"] = [election_label_to_key[label] for label in selected_labels if label in election_label_to_key]
        selected["선거명"] = election_candidates.loc[
            election_candidates["label"].astype("string").isin(selected_labels),
            "선거명",
        ].dropna().astype("string").tolist()
        selected["선거종류"] = election_candidates.loc[
            election_candidates["label"].astype("string").isin(selected_labels),
            "선거종류",
        ].dropna().astype("string").tolist()

    dynamic_frame = _filter_by_selected(common_base, selected)
    for column in HIERARCHY_FILTER_COLUMNS:
        options = _sorted_options(dynamic_frame, column)
        _sanitize_widget_state(FILTER_WIDGET_KEYS[column], options)
        sidebar.multiselect(column, options=options, key=FILTER_WIDGET_KEYS[column])
        selected[column] = _current_selection(column)
        if selected[column]:
            dynamic_frame = dynamic_frame.loc[dynamic_frame[column].astype("string").isin(selected[column])]

    adjacent_options: list[str] = []
    auto_adjacent_names: list[str] = []
    focus_gusigun_key: str | None = None
    focus_sido_name: str | None = None
    if len(selected["구시군명"]) == 1 and not gusigun_lookup.empty:
        focus_rows = gusigun_lookup.loc[gusigun_lookup["구시군명"].astype("string") == str(selected["구시군명"][0])].copy()
        if selected.get("시도명"):
            focus_rows = focus_rows.loc[focus_rows["시도명"].astype("string").isin(selected["시도명"])].copy()
        focus_keys = focus_rows["구시군KEY"].dropna().astype("string").unique().tolist()
        if len(focus_keys) == 1:
            focus_gusigun_key = focus_keys[0]
            focus_sido_values = focus_rows["시도명"].dropna().astype("string").unique().tolist()
            focus_sido_name = focus_sido_values[0] if focus_sido_values else None
            if focus_sido_name:
                same_sido_lookup = gusigun_lookup.loc[gusigun_lookup["시도명"].astype("string") == str(focus_sido_name)].copy()
                adjacent_options = sorted(
                    [
                        name
                        for name in same_sido_lookup["구시군명"].dropna().astype("string").unique().tolist()
                        if name != str(selected["구시군명"][0])
                    ]
                )
                from src.maps import get_adjacent_gusigun_keys

                auto_adjacent_keys = set(get_adjacent_gusigun_keys(focus_gusigun_key))
                auto_adjacent_names = sorted(
                    [
                        name
                        for name in same_sido_lookup.loc[
                            same_sido_lookup["구시군KEY"].astype("string").isin(auto_adjacent_keys),
                            "구시군명",
                        ].dropna().astype("string").unique().tolist()
                        if name != str(selected["구시군명"][0])
                    ]
                )

    if focus_gusigun_key:
        if st.session_state.get(ADJACENT_SOURCE_STATE_KEY) != focus_gusigun_key:
            st.session_state[FILTER_WIDGET_KEYS["인접구시군명"]] = auto_adjacent_names
            st.session_state[ADJACENT_SOURCE_STATE_KEY] = focus_gusigun_key
    else:
        st.session_state[FILTER_WIDGET_KEYS["인접구시군명"]] = []
        st.session_state.pop(ADJACENT_SOURCE_STATE_KEY, None)

    _sanitize_widget_state(FILTER_WIDGET_KEYS["인접구시군명"], adjacent_options)
    sidebar.multiselect("인접 구시군권", options=adjacent_options, key=FILTER_WIDGET_KEYS["인접구시군명"])
    selected["인접구시군명"] = _current_selection("인접구시군명")
    if focus_gusigun_key and focus_sido_name:
        sidebar.caption(f"인접 구시군권은 기본으로 자동 선택되며, 필요하면 {focus_sido_name} 내 다른 구시군도 추가/제거할 수 있습니다.")
    else:
        sidebar.caption("구시군을 1곳 선택하면 인접 구시군권을 자동으로 채워 드립니다.")

    if include_party:
        party_candidates = _filter_by_selected(party_base, selected)
        party_options = _sorted_options(party_candidates, "정당명")
        _sanitize_widget_state(FILTER_WIDGET_KEYS["정당명"], party_options)
        sidebar.multiselect("정당명", options=party_options, key=FILTER_WIDGET_KEYS["정당명"])
        selected["정당명"] = _current_selection("정당명")
    else:
        selected["정당명"] = _current_selection("정당명")

    if include_candidate:
        candidate_candidates = _filter_by_selected(candidate_base, selected)
        candidate_options = _sorted_options(candidate_candidates, "후보명")
        _sanitize_widget_state(FILTER_WIDGET_KEYS["후보명"], candidate_options)
        sidebar.multiselect("후보명", options=candidate_options, key=FILTER_WIDGET_KEYS["후보명"])
        selected["후보명"] = _current_selection("후보명")
    else:
        selected["후보명"] = _current_selection("후보명")

    if show_election_filters:
        sidebar.caption("다중 선택된 선거, 지역, 정당, 후보는 페이지 이동 후에도 유지됩니다.")

    return selected


def apply_common_filters(df: pd.DataFrame, selected: dict[str, list[str]]) -> pd.DataFrame:
    result = df

    if selected.get("선거KEY"):
        election_keys = [str(value) for value in selected["선거KEY"]]
        election_periods = [value[:6] for value in election_keys]
        if "선거KEY" in result.columns:
            result = result.loc[result["선거KEY"].astype("string").isin(election_keys)]
        elif "선거시점" in result.columns:
            result = result.loc[result["선거시점"].astype("string").str[:6].isin(election_periods)]
    elif selected.get("선거시점"):
        periods = [str(value) for value in selected["선거시점"]]
        if "선거시점" in result.columns:
            result = result.loc[result["선거시점"].astype("string").str[:6].isin(periods)]
        elif "선거KEY" in result.columns:
            result = result.loc[result["선거KEY"].astype("string").str[:6].isin(periods)]

    for canonical_column in HIERARCHY_FILTER_COLUMNS + ENTITY_FILTER_COLUMNS:
        selected_values = selected.get(canonical_column)
        if not selected_values:
            continue

        candidate_columns = [column for column in FILTER_ALIASES[canonical_column] if column in result.columns]
        if not candidate_columns:
            continue

        result = result.loc[result[candidate_columns[0]].astype("string").isin(selected_values)]

    return result


def summarize_filter_selection(
    selected: dict[str, list[str]],
    filter_options: dict[str, object],
) -> str:
    labels: list[str] = []
    election_key_to_label = filter_options["election_key_to_label"]

    if selected.get("선거KEY"):
        labels.append("선거: " + ", ".join(election_key_to_label.get(key, key) for key in selected["선거KEY"]))
    elif selected.get("선거시점"):
        labels.append("선거시점: " + ", ".join(selected["선거시점"]))

    for column in HIERARCHY_FILTER_COLUMNS + ENTITY_FILTER_COLUMNS:
        if selected.get(column):
            labels.append(f"{column}: " + ", ".join(selected[column]))

    return " / ".join(labels) if labels else "전체"


def build_breadcrumb_text(
    selected: dict[str, list[str]],
    filter_options: dict[str, object],
) -> str:
    return summarize_filter_selection(selected, filter_options)


def resolve_region_level(selected: dict[str, list[str]], default: str = "구시군") -> str:
    if selected.get("읍면동명"):
        return "읍면동"
    if selected.get("구시군명") or selected.get("일반구명"):
        return "구시군"
    if selected.get("시도명"):
        return "시도"
    return default


def clear_entity_filters(selected: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    result["정당명"] = []
    result["후보명"] = []
    return result


def clear_rowtype_filter(selected: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    result["RowType"] = []
    return result


def build_region_comparison_selection(selected: dict[str, list[str]], level: str) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    normalized_level = level.strip()

    if normalized_level == "구시군":
        result["구시군명"] = []
        result["일반구명"] = []
        result["읍면동명"] = []
    elif normalized_level == "읍면동":
        result["읍면동명"] = []
    elif normalized_level == "시도":
        result["구시군명"] = []
        result["일반구명"] = []
        result["읍면동명"] = []

    return result


def build_scope_selection(selected: dict[str, list[str]], scope: str) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    normalized_scope = scope.strip().lower()

    if normalized_scope == "national":
        result["시도명"] = []
        result["구시군명"] = []
        result["일반구명"] = []
        result["읍면동명"] = []
        return result

    if normalized_scope == "sido":
        result["구시군명"] = []
        result["일반구명"] = []
        result["읍면동명"] = []
        return result

    if normalized_scope == "gusigun":
        result["일반구명"] = []
        result["읍면동명"] = []
        return result

    return result


def count_selected_regions(selected: dict[str, list[str]]) -> int:
    for column in ["읍면동명", "구시군명", "시도명"]:
        if selected.get(column):
            return len(selected[column])
    return 0


def get_selected_adjacent_gusigun_names(selected: dict[str, list[str]]) -> list[str]:
    return [str(value) for value in selected.get("인접구시군명", []) if value not in (None, "")]
