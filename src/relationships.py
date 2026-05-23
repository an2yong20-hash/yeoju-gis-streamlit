from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.qa import check_null_keys, check_required_columns, validate_key_coverage, validate_relationship


@dataclass(frozen=True)
class RelationshipSpec:
    fact_name: str
    fact_key: str
    dim_name: str
    dim_key: str


def get_relationship_specs() -> dict[str, list[RelationshipSpec]]:
    return {
        "FactResidentComposition": [
            RelationshipSpec("FactResidentComposition", "구시군KEY", "DimGusigun", "구시군KEY"),
            RelationshipSpec("FactResidentComposition", "선거KEY", "DimElection", "선거KEY"),
            RelationshipSpec("FactResidentComposition", "읍면동KEY", "DimDong", "읍면동KEY"),
        ],
        "FactConfirmedElectorate": [
            RelationshipSpec("FactConfirmedElectorate", "구시군KEY", "DimGusigun", "구시군KEY"),
            RelationshipSpec("FactConfirmedElectorate", "선거KEY", "DimElection", "선거KEY"),
            RelationshipSpec("FactConfirmedElectorate", "읍면동KEY", "DimDong", "읍면동KEY"),
            RelationshipSpec("FactConfirmedElectorate", "투표소KEY", "DimPollingPlace", "투표소KEY"),
        ],
        "FactTurnout": [
            RelationshipSpec("FactTurnout", "구시군KEY", "DimGusigun", "구시군KEY"),
            RelationshipSpec("FactTurnout", "선거KEY", "DimElection", "선거KEY"),
            RelationshipSpec("FactTurnout", "읍면동KEY", "DimDong", "읍면동KEY"),
            RelationshipSpec("FactTurnout", "투표소KEY", "DimPollingPlace", "투표소KEY"),
        ],
        "FactVotes": [
            RelationshipSpec("FactVotes", "구시군KEY", "DimGusigun", "구시군KEY"),
            RelationshipSpec("FactVotes", "선거KEY", "DimElection", "선거KEY"),
            RelationshipSpec("FactVotes", "읍면동KEY", "DimDong", "읍면동KEY"),
            RelationshipSpec("FactVotes", "정당KEY", "DimParty", "정당KEY"),
            RelationshipSpec("FactVotes", "투표소KEY", "DimPollingPlace", "투표소KEY"),
        ],
    }


def attach_dimensions_to_fact(
    fact_df: pd.DataFrame,
    dims: dict[str, pd.DataFrame],
    fact_name: str,
) -> pd.DataFrame:
    relationship_specs = get_relationship_specs()
    if fact_name not in relationship_specs:
        raise KeyError(f"Unknown fact name for relationship attachment: {fact_name}")

    attached = fact_df.copy()
    for spec in relationship_specs[fact_name]:
        if spec.dim_name not in dims:
            raise KeyError(f"Missing dimension table '{spec.dim_name}' for {fact_name}")

        dim_df = dims[spec.dim_name]
        check_required_columns(attached, [spec.fact_key], fact_name)
        check_required_columns(dim_df, [spec.dim_key], spec.dim_name)

        prefixed_dim = dim_df.add_prefix(f"{spec.dim_name}__")
        right_key = f"{spec.dim_name}__{spec.dim_key}"
        attached = attached.merge(prefixed_dim, left_on=spec.fact_key, right_on=right_key, how="left")

    return attached


def validate_all_relationships(
    facts: dict[str, pd.DataFrame],
    dims: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict[str, object]]]:
    results: dict[str, dict[str, dict[str, object]]] = {}

    for fact_name, specs in get_relationship_specs().items():
        if fact_name not in facts:
            continue

        fact_df = facts[fact_name]
        fact_results: dict[str, dict[str, object]] = {}

        for spec in specs:
            if spec.dim_name not in dims:
                raise KeyError(f"Missing dimension table '{spec.dim_name}' required by {fact_name}")

            dim_df = dims[spec.dim_name]
            relationship_name = f"{spec.fact_key}->{spec.dim_name}[{spec.dim_key}]"

            null_key_rows = check_null_keys(fact_df, [spec.fact_key], fact_name)
            coverage = validate_key_coverage(
                fact_df=fact_df,
                dim_df=dim_df,
                fact_key=spec.fact_key,
                dim_key=spec.dim_key,
                name=relationship_name,
            )
            relationship = validate_relationship(
                left_df=fact_df,
                right_df=dim_df,
                left_key=spec.fact_key,
                right_key=spec.dim_key,
                left_name=fact_name,
                right_name=spec.dim_name,
            )

            fact_results[relationship_name] = {
                "spec": asdict(spec),
                "null_key_rows": int(len(null_key_rows)),
                "coverage": coverage,
                "relationship": relationship,
            }

        results[fact_name] = fact_results

    return results
