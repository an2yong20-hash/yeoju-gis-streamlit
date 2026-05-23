from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "DB"
DIM_DIR = DB_DIR / "Dim"
FACT_DIR = DB_DIR / "Fact"
FACT_VOTES_DIR = FACT_DIR / "FactVotes"
FACT_CONFIRMED_ELECTORATE_PATH = FACT_DIR / "FactConfirmedElectorate.csv"
FACT_RESIDENT_COMPOSITION_PATH = FACT_DIR / "FactResidentComposition.csv"
CACHE_DIR = BASE_DIR / "cache"

RAW_ENCODING = "cp949"
DEFAULT_ENCODING = RAW_ENCODING

SIDO_NAME_STANDARDIZATION = {
    "강원도": "강원특별자치도",
    "전라북도": "전북특별자치도",
    "세종특별시": "세종특별자치시",
}
