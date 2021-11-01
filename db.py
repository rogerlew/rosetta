'''
    Rosetta version 3-alpha (3a) 
    Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
    Copyright (C) 2016  Marcel G. Schaap

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


import os
import sys
import sqlite3

from os.path import join as _join
from os.path import exists

_this_dir = os.path.dirname(__file__)
_sqlite_path = _join(_this_dir, "sqlite/Rosetta.sqlite")

class DB(object):

    @property
    def conn(self): 
        return(self._conn)

    def __init__(self, debug=False):
        self.debug = debug

        if not exists(_sqlite_path):
            raise Exception("Cannot find the sqlite path '%s'" % _sqlite_path)
            self.Error = sqlite3.Error

        try:
            self._conn = sqlite3.connect(sqlite_path)
        except self.Error as e:
            raise Exception("Database connection error %d: %s" % (e.args[0], e.args[1]))

        self.conn.text_factory = bytes
        self.sqlite = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.commit()
        self.close()
        if self.debug:
            print("Closed DB object")

    def get_cursor(self):
        try:
            return  self.conn.cursor()
        except self.Error as e:
            raise Exception("Database cursor error %d: %s" % (e.args[0], e.args[1]))

    def commit(self): 
        if self.conn:
           self.conn.commit()

    def close(self):  
        if self.conn: 
            self.conn.close()

