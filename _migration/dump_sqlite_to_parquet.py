#!/usr/bin/env python3
"""Dump every SQLite table to individual Parquet files."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SQLITE = Path(__file__).resolve().parents[1] / "sqlite" / "Rosetta.sqlite"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "db"


def iter_table_names(conn: sqlite3.Connection) -> Iterable[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    for (name,) in cursor.fetchall():
        if isinstance(name, bytes):
            name = name.decode()
        yield name


def export_table(conn: sqlite3.Connection, table: str, output_dir: Path) -> Path:
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    output_path = output_dir / f"{table}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def main(sqlite_path: Path, output_dir: Path, tables: Iterable[str] | None = None) -> None:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as conn:
        # Ensure bytes are not implicitly decoded as text
        conn.text_factory = bytes
        available_tables = list(iter_table_names(conn))

        if tables:
            missing = sorted(set(tables) - set(available_tables))
            if missing:
                raise ValueError(f"Requested tables not found: {', '.join(missing)}")
            target_tables = list(tables)
        else:
            target_tables = available_tables

        if not target_tables:
            raise ValueError("No tables selected for export")

        print(f"Exporting {len(target_tables)} tables from {sqlite_path} to {output_dir}")

        for table in target_tables:
            path = export_table(conn, table, output_dir)
            print(f"  wrote {table} -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=DEFAULT_SQLITE,
        help=f"Path to the SQLite database (default: {DEFAULT_SQLITE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store Parquet exports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "tables",
        nargs="*",
        help="Optional list of specific tables to export",
    )
    args = parser.parse_args()

    main(args.sqlite, args.out, args.tables or None)
