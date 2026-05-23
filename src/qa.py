from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _as_key_list(keys: str | Iterable[str]) -> list[str]:
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def _normalize_key_component(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _build_row_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    check_required_columns(df, key_cols, "key frame")

    if len(key_cols) == 1:
        key_col = key_cols[0]
        return df[key_col].map(_normalize_key_component)

    return df[key_cols].apply(
        lambda row: (
            None
            if any(_normalize_key_component(value) is None for value in row)
            else tuple(_normalize_key_component(value) for value in row)
        ),
        axis=1,
    )


def check_required_columns(df: pd.DataFrame, required_cols: Iterable[str], name: str) -> None:
    required_cols = list(required_cols)
    missing_columns = [column for column in required_cols if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{name}: missing required columns: {missing_columns}")


def check_null_keys(df: pd.DataFrame, key_cols: Iterable[str], name: str) -> pd.DataFrame:
    key_cols = list(key_cols)
    check_required_columns(df, key_cols, name)

    null_mask = pd.Series(False, index=df.index)
    for column in key_cols:
        column_mask = df[column].isna()
        if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            column_mask = column_mask | df[column].astype("string").str.strip().eq("")
        null_mask = null_mask | column_mask

    return df.loc[null_mask, key_cols].copy()


def check_duplicate_keys(df: pd.DataFrame, key_cols: Iterable[str], name: str) -> pd.DataFrame:
    key_cols = list(key_cols)
    check_required_columns(df, key_cols, name)

    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    if not duplicate_mask.any():
        return df.iloc[0:0].copy()

    return df.loc[duplicate_mask].sort_values(by=key_cols, kind="stable").copy()


def summarize_dataframe(df: pd.DataFrame, name: str) -> dict[str, object]:
    return {
        "name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(count) for column, count in df.isna().sum().items()},
    }


def validate_relationship(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str | Iterable[str],
    right_key: str | Iterable[str],
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    left_key_cols = _as_key_list(left_key)
    right_key_cols = _as_key_list(right_key)

    check_required_columns(left_df, left_key_cols, left_name)
    check_required_columns(right_df, right_key_cols, right_name)

    left_keys = _build_row_keys(left_df, left_key_cols)
    right_keys = _build_row_keys(right_df, right_key_cols)

    left_valid = left_keys[left_keys.notna()]
    right_valid = right_keys[right_keys.notna()]

    left_set = set(left_valid.tolist())
    right_set = set(right_valid.tolist())
    unmatched = sorted(left_set - right_set, key=str)

    return {
        "left_name": left_name,
        "right_name": right_name,
        "left_key": left_key_cols,
        "right_key": right_key_cols,
        "left_distinct_keys": len(left_set),
        "right_distinct_keys": len(right_set),
        "matched_distinct_keys": len(left_set & right_set),
        "unmatched_distinct_keys": len(unmatched),
        "unmatched_rate": (len(unmatched) / len(left_set)) if left_set else 0.0,
        "sample_unmatched_keys": unmatched[:20],
    }


def validate_rowtype_distribution(df: pd.DataFrame, rowtype_col: str = "RowType") -> pd.DataFrame:
    check_required_columns(df, [rowtype_col], "rowtype frame")

    distribution = (
        df[rowtype_col]
        .fillna("<NA>")
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis(rowtype_col)
        .reset_index(name="count")
    )
    distribution["ratio"] = distribution["count"] / len(df) if len(df) else 0.0
    return distribution


def validate_key_coverage(
    fact_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    fact_key: str | Iterable[str],
    dim_key: str | Iterable[str],
    name: str,
) -> dict[str, object]:
    fact_key_cols = _as_key_list(fact_key)
    dim_key_cols = _as_key_list(dim_key)

    check_required_columns(fact_df, fact_key_cols, name)
    check_required_columns(dim_df, dim_key_cols, f"{name} reference")

    fact_keys = _build_row_keys(fact_df, fact_key_cols)
    dim_keys = set(_build_row_keys(dim_df, dim_key_cols).dropna().tolist())

    valid_fact_mask = fact_keys.notna()
    covered_mask = valid_fact_mask & fact_keys.isin(dim_keys)
    uncovered_keys = sorted(set(fact_keys[valid_fact_mask & ~fact_keys.isin(dim_keys)].tolist()), key=str)

    valid_rows = int(valid_fact_mask.sum())
    covered_rows = int(covered_mask.sum())

    return {
        "name": name,
        "fact_key": fact_key_cols,
        "dim_key": dim_key_cols,
        "valid_fact_rows": valid_rows,
        "covered_rows": covered_rows,
        "uncovered_rows": valid_rows - covered_rows,
        "coverage_rate": (covered_rows / valid_rows) if valid_rows else 0.0,
        "sample_uncovered_keys": uncovered_keys[:20],
    }


def generate_data_quality_report(
    facts: dict[str, pd.DataFrame],
    dims: dict[str, pd.DataFrame],
) -> dict[str, object]:
    report: dict[str, object] = {
        "facts": {},
        "dims": {},
        "rowtype_distribution": {},
        "relationships": {},
        "coverage": {},
    }

    for name, dataframe in dims.items():
        report["dims"][name] = summarize_dataframe(dataframe, name)

    for name, dataframe in facts.items():
        report["facts"][name] = summarize_dataframe(dataframe, name)
        if "RowType" in dataframe.columns:
            report["rowtype_distribution"][name] = validate_rowtype_distribution(dataframe).to_dict(
                orient="records"
            )

    if "FactTurnout" in facts and "DimDong" in dims:
        report["coverage"]["FactTurnout->DimDong"] = validate_key_coverage(
            facts["FactTurnout"],
            dims["DimDong"],
            "읍면동KEY",
            "읍면동KEY",
            "FactTurnout",
        )

    if "FactConfirmedElectorate" in facts and "DimDong" in dims:
        report["coverage"]["FactConfirmedElectorate->DimDong"] = validate_key_coverage(
            facts["FactConfirmedElectorate"],
            dims["DimDong"],
            "읍면동KEY",
            "읍면동KEY",
            "FactConfirmedElectorate",
        )

    if "FactResidentComposition" in facts and "DimDong" in dims:
        report["coverage"]["FactResidentComposition->DimDong"] = validate_key_coverage(
            facts["FactResidentComposition"],
            dims["DimDong"],
            "읍면동KEY",
            "읍면동KEY",
            "FactResidentComposition",
        )

    if "FactVotes" in facts and "DimDong" in dims:
        report["coverage"]["FactVotes->DimDong"] = validate_key_coverage(
            facts["FactVotes"],
            dims["DimDong"],
            "읍면동KEY",
            "읍면동KEY",
            "FactVotes",
        )

    if "FactVotes" in facts and "DimParty" in dims:
        report["coverage"]["FactVotes->DimParty"] = validate_key_coverage(
            facts["FactVotes"],
            dims["DimParty"],
            "정당KEY",
            "정당KEY",
            "FactVotes",
        )
        report["relationships"]["FactVotes->DimParty"] = validate_relationship(
            facts["FactVotes"],
            dims["DimParty"],
            "정당KEY",
            "정당KEY",
            "FactVotes",
            "DimParty",
        )

    return report


def compare_fact_dim_counts(
    facts: dict[str, pd.DataFrame],
    dims: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    from src.relationships import get_relationship_specs

    rows: list[dict[str, object]] = []
    for fact_name, specs in get_relationship_specs().items():
        if fact_name not in facts:
            continue

        fact_df = facts[fact_name]
        for spec in specs:
            if spec.dim_name not in dims:
                continue

            dim_df = dims[spec.dim_name]
            check_required_columns(fact_df, [spec.fact_key], fact_name)
            check_required_columns(dim_df, [spec.dim_key], spec.dim_name)

            fact_keys = _build_row_keys(fact_df, [spec.fact_key])
            dim_keys = _build_row_keys(dim_df, [spec.dim_key])
            fact_distinct = set(fact_keys.dropna().tolist())
            dim_distinct = set(dim_keys.dropna().tolist())

            rows.append(
                {
                    "fact_name": fact_name,
                    "fact_key": spec.fact_key,
                    "dim_name": spec.dim_name,
                    "dim_key": spec.dim_key,
                    "fact_rows": int(len(fact_df)),
                    "fact_distinct_keys": len(fact_distinct),
                    "dim_rows": int(len(dim_df)),
                    "dim_distinct_keys": len(dim_distinct),
                    "matched_distinct_keys": len(fact_distinct & dim_distinct),
                    "unmatched_distinct_keys": len(fact_distinct - dim_distinct),
                }
            )

    return pd.DataFrame(rows)


def summarize_unmatched_keys(
    relationship_results: dict[str, dict[str, dict[str, object]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fact_name, result_map in relationship_results.items():
        for relationship_name, result in result_map.items():
            relationship = result["relationship"]
            coverage = result["coverage"]
            rows.append(
                {
                    "fact_name": fact_name,
                    "relationship": relationship_name,
                    "null_key_rows": result["null_key_rows"],
                    "unmatched_distinct_keys": relationship["unmatched_distinct_keys"],
                    "unmatched_rate": relationship["unmatched_rate"],
                    "uncovered_rows": coverage["uncovered_rows"],
                    "coverage_rate": coverage["coverage_rate"],
                    "sample_unmatched_keys": " | ".join(str(key) for key in relationship["sample_unmatched_keys"]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fact_name",
                "relationship",
                "null_key_rows",
                "unmatched_distinct_keys",
                "unmatched_rate",
                "uncovered_rows",
                "coverage_rate",
                "sample_unmatched_keys",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["unmatched_distinct_keys", "uncovered_rows"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def summarize_duplicates(
    tables: dict[str, pd.DataFrame],
    key_map: dict[str, str | Iterable[str]] | None = None,
) -> pd.DataFrame:
    default_key_map: dict[str, str | list[str]] = {
        "DimDong": "읍면동KEY",
        "DimGusigun": "구시군KEY",
        "DimElection": "선거KEY",
        "DimParty": "정당KEY",
        "DimPollingPlace": "투표소KEY",
        "FactConfirmedElectorate": ["선거KEY", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "투표구명"],
        "FactResidentComposition": ["선거KEY", "RowType", "행정기관코드"],
        "FactTurnout": ["선거KEY", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "투표구명"],
        "FactVotes": ["선거KEY", "RowType", "정당KEY", "구시군KEY", "읍면동KEY", "투표소KEY", "후보명", "구분"],
    }
    key_map = default_key_map if key_map is None else key_map

    rows: list[dict[str, object]] = []
    for table_name, keys in key_map.items():
        if table_name not in tables:
            continue

        key_cols = _as_key_list(keys)
        table_df = tables[table_name]
        check_required_columns(table_df, key_cols, table_name)

        null_rows = check_null_keys(table_df, key_cols, table_name)
        valid_df = table_df.drop(index=null_rows.index, errors="ignore")
        duplicates = check_duplicate_keys(valid_df, key_cols, table_name)

        rows.append(
            {
                "table_name": table_name,
                "key_cols": ", ".join(key_cols),
                "rows": int(len(table_df)),
                "null_key_rows": int(len(null_rows)),
                "duplicate_rows": int(len(duplicates)),
                "sample_duplicate_keys": ""
                if duplicates.empty
                else " | ".join(
                    str(value)
                    for value in duplicates.loc[:, key_cols].drop_duplicates().head(5).itertuples(index=False, name=None)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["duplicate_rows", "null_key_rows"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def build_validation_snapshot(
    facts: dict[str, pd.DataFrame],
    dims: dict[str, pd.DataFrame],
) -> dict[str, object]:
    from src.relationships import validate_all_relationships

    relationship_results = validate_all_relationships(facts, dims)
    quality_report = generate_data_quality_report(facts, dims)
    count_comparison = compare_fact_dim_counts(facts, dims)
    unmatched_keys = summarize_unmatched_keys(relationship_results)
    duplicates = summarize_duplicates({**dims, **facts})

    return {
        "quality_report": quality_report,
        "relationship_results": relationship_results,
        "count_comparison": count_comparison,
        "unmatched_keys": unmatched_keys,
        "duplicates": duplicates,
    }
