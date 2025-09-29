'''
    Rosetta version 3-alpha (3a) 
    Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
    Copyright (C) 2016  Marcel G. Schaap
    Copyright (C) 2021  Roger Lew <rogerlew@gmail.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Marcel G. Schaap can be contacted at:
    mschaap@cals.arizona.edu

'''


from pathlib import Path

import duckdb

_this_dir = Path(__file__).resolve().parent
_duckdb_path = _this_dir / "db" / "rosetta.duckdb"


def _convert_row(values):
    converted = []
    for value in values:
        if isinstance(value, str):
            converted.append(value.encode("utf-8"))
        elif isinstance(value, memoryview):
            converted.append(bytes(value))
        else:
            converted.append(value)
    return tuple(converted)


class DuckCursor:

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._relation = None
        self._rows = None

    def execute(self, sql, params=None):
        if params is None:
            self._relation = self._conn.execute(sql)
        else:
            self._relation = self._conn.execute(sql, params)
        self._rows = None
        return self

    def _ensure_rows(self):
        if self._rows is None and self._relation is not None:
            fetched = self._relation.fetchall()
            self._rows = [_convert_row(row) for row in fetched]
        return self._rows or []

    def fetchall(self):
        return list(self._ensure_rows())

    def fetchone(self):
        rows = self._ensure_rows()
        return rows[0] if rows else None

    def __iter__(self):
        return iter(self._ensure_rows())

    def close(self):
        self._relation = None
        self._rows = None


class DB(object):

    @property
    def conn(self):
        return self._conn

    @property
    def readonly(self):
        return self._readonly

    def __init__(self, debug=False, readonly=True, database_path: Path | None = None):
        self.debug = debug
        self._readonly = readonly

        db_path = Path(database_path) if database_path else _duckdb_path
        if not db_path.exists():
            raise Exception(f"Cannot find the DuckDB database '{db_path}'")

        self._conn = duckdb.connect(str(db_path), read_only=readonly)
        self.duckdb = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.readonly:
            self.commit()
        self.close()
        if self.debug:
            print("Closed DB object")

    def get_cursor(self):
        return DuckCursor(self.conn)

    def commit(self):
        if self.readonly:
            raise Exception('DB was opened readonly ')

        if self.conn:
            self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
