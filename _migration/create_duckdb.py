#!/usr/bin/env python3
"""Create a DuckDB database from the exported Parquet tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import duckdb

DEFAULT_PARQUET_DIR = Path(__file__).resolve().parents[1] / "db"
DEFAULT_DUCKDB = DEFAULT_PARQUET_DIR / "rosetta.duckdb"


def discover_parquet_tables(parquet_dir: Path) -> Iterable[Path]:
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    yield from sorted(parquet_dir.glob("*.parquet"))


def create_database(database_path: Path, parquet_dir: Path, replace: bool = False) -> None:
    parquet_files = list(discover_parquet_tables(parquet_dir))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    if replace and database_path.exists():
        database_path.unlink()

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(database_path)) as conn:
        for parquet_file in parquet_files:
            table_name = parquet_file.stem
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.execute(
                f'CREATE TABLE "{table_name}" AS SELECT * FROM read_parquet(?)',
                [str(parquet_file)],
            )
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            print(f"  loaded {table_name} ({count} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help=f"Directory containing parquet tables (default: {DEFAULT_PARQUET_DIR})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DUCKDB,
        help=f"Output DuckDB database path (default: {DEFAULT_DUCKDB})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite the database if it already exists",
    )
    args = parser.parse_args()

    create_database(args.database, args.parquet_dir, replace=args.replace)


if __name__ == "__main__":
    main()
