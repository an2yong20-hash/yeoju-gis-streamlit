from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _data_base_dir(root: Path) -> Path:
    if not getattr(sys, "frozen", False) and (root / "gis_data" / "cache").exists():
        return root / "gis_data"
    if (root / "cache").exists():
        return root
    if (root / "gis_data" / "cache").exists():
        return root / "gis_data"
    return root


def _install_config_module(base_dir: Path) -> None:
    config = types.ModuleType("config")
    config.BASE_DIR = base_dir
    config.DB_DIR = base_dir / "DB"
    config.DIM_DIR = config.DB_DIR / "Dim"
    config.FACT_DIR = config.DB_DIR / "Fact"
    config.FACT_VOTES_DIR = config.FACT_DIR / "FactVotes"
    config.FACT_CONFIRMED_ELECTORATE_PATH = config.FACT_DIR / "FactConfirmedElectorate.csv"
    config.FACT_RESIDENT_COMPOSITION_PATH = config.FACT_DIR / "FactResidentComposition.csv"
    config.CACHE_DIR = base_dir / "cache"
    config.RAW_ENCODING = "cp949"
    config.DEFAULT_ENCODING = config.RAW_ENCODING
    config.SIDO_NAME_STANDARDIZATION = {
        "강원도": "강원특별자치도",
        "전라북도": "전북특별자치도",
        "세종특별시": "세종특별자치시",
    }
    sys.modules["config"] = config


def _run_gis_page() -> None:
    root = _resource_root()
    sys.path.insert(0, str(root))
    _install_config_module(_data_base_dir(root))

    bundled_home = root / "gis_data" / "home"
    if bundled_home.exists():
        Path.home = classmethod(lambda cls: bundled_home)

    page_path = root / "pages" / "7_GIS분석.py"
    if not page_path.exists():
        raise FileNotFoundError(f"GIS Streamlit page not found: {page_path}")
    runpy.run_path(str(page_path), run_name="__main__")


_run_gis_page()
