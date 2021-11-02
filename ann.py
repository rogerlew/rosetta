"""
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
"""

import os
import sys
import io
import struct
import gzip
import hashlib
from contextlib import closing
from functools import reduce

import numpy as np


class ANN(object):

    @property
    def index(self):
        return self._cnf[0]

    @index.setter  # we might need this because we could renumber the index
    def index(self, value):
        self._cnf[0] = value

    @property
    def nin(self):
        return self._cnf[1]

    @property
    def nlayer(self):
        return self._cnf[2]

    @nlayer.setter
    def nlayer(self, val):
        self._cnf[2] = val  # hack to read old rosetta

    @property
    def nhid1(self):
        return self._cnf[3]

    @property
    def nhid2(self):
        return self._cnf[4]

    @property
    def nout(self):
        return self._cnf[5]

    @property
    def hash_id(self):
        return self._hash_i

    @hash_id.setter
    def hash_id(self, value):
        self._hash_id = value

    @property
    def model_id(self):
        return self._model

    @model_id.setter
    def model_id(self, value):
        self._model = value

    @property
    def cnf(self):
        return self._cnf

    @cnf.setter
    def cnf(self, cnf):
        self._cnf = cnf

    @property
    def transfer_funcs(self):
        return self._transfer_funcs

    @transfer_funcs.setter
    def transfer_funcs(self, val):
        self._transfer_funcs = val

    def rd_hash(self, fbin):
        hash_id = fbin.read(32)
        assert len(hash_id) == 32, 'len(hash_id <> 32'
        model = struct.unpack('i', fbin.read(4))[0]
        return hash_id, model

    def read_mat(self, fbin, import_bin=False):
        """
        so rows goes faster in ML, whereas COLS goes faster in PYTHON (and C)
        do we need to convert upon import to DB?
        """
        size = (struct.unpack('i', fbin.read(4))[
                0], struct.unpack('i', fbin.read(4))[0])
        assert size[0] < 100, size
        assert size[1] < 100, size
        mat = np.zeros(size)
        if import_bin:
            # columns change fastest (Matlab, order=F)
            for j in range(size[1]):
                for i in range(size[0]):
                    mat[i, j] = struct.unpack('d', fbin.read(8))[0]
        else:
            # row change fastest (Python, C)
            for j in range(size[0]):
                for i in range(size[1]):
                    mat[j, i] = struct.unpack('d', fbin.read(8))[0]

        return mat

    def parse_transfer_funcs(self, nl, tr):
        nl = int(nl)
        tr = [x.decode('utf-8') for x in tr]
        if tr is None:
            return [self.tansig, self.logsig]  # Yonggen

        ntr = len(tr)
        assert ntr in [2, 3], 'number of transfer functions must be 2 or 3'
        assert nl in [2, 3], 'number of layers must be 2 or 3'
        assert nl <= ntr, 'number of layers must be smaller or equal to number of transfer functions'

        # the problem is that currently nl=2 but ntr=3 (from DB)
        if nl == 2 and ntr == 3:
            trhelp = tr[:1] + tr[-1:]
        else:  # (nl=2 and ntr=2) OR (nl=3 and ntr=3)
            trhelp = tr

        funcs = []
        funcs_names = []
        for i in range(nl):
            _trhelp_lower = trhelp[i].lower()
            if _trhelp_lower == 'tansig':
                funcs.append(self.tansig)
                funcs_names.append('tansig')
            elif _trhelp_lower == 'logsig':
                funcs.append(self.logsig)
                funcs_names.append('logsig')
            elif _trhelp_lower== ' purelin':
                funcs.append(self.purelin)
                funcs_names.append('purelin')
            elif _trhelp_lower == 'none':
                funcs.append(None)
                funcs_names.append('none')
            else:
                raise Exception('Unknown transfer function')
        return funcs, funcs_names

    def __init__(self, nlayer=2, transfers=None):
        self.w = []
        self.b = []

        self.transfer_funcs, self.transfer_names = self.parse_transfer_funcs(nlayer, transfers)

    def read(self, fbin, import_bin=False, oldrosetta=False, hash_id=None, model=None):
        '''
        Note: all the numpy fromfiles were removed and replaced with more complex code
        because numpy expects fbin to be a real file, whereas we also need to be a string 
        that has been buffered by (c)StringIO
        '''
        assert fbin, 'fbin not open in ann.__init__'

        count = 6
        self.cnf = np.zeros((count,), dtype=np.int32)

        if not oldrosetta:
            self.hash_id, self.model = self.rd_hash(fbin)
        else:
            self.hash_id, self.model = hash_id, model

        for i in range(count):
            self.cnf[i] = struct.unpack('i', fbin.read(4))[0]
        # Old rosetta gives nhidden, not nlayer!
        if oldrosetta:
            self.nlayer += 1

        # print(self.cnf)
        for i in range(self.nlayer):
            self.w.append(self.read_mat(fbin, import_bin))  # weights
            self.b.append(self.read_mat(fbin, import_bin))  # biases

    @staticmethod
    def from_stream(fbin, import_bin=False, oldrosetta=False, hash_id=None, model_id=None, nlayer=None, transfers=None):
        # nlayer and transfers is NOT in the binary file, so need to pull from somewhere else!
        ann = ANN(nlayer=nlayer, transfers=transfers)
        ann.read(fbin, import_bin, oldrosetta, hash_id, model_id)
        return ann

    def tostring(self):
        s = self.hash_id
        s += np.array([self.model], dtype=np.int32).tostring()  # could be byte
        s += self.cnf.tostring()
        for i in range(self.nlayer):
            s += np.array(np.shape(self.w[i]),
                          dtype=np.int32).tostring()  # rows, cols
            s += self.w[i].tostring(order='C')  # the array itself
            s += np.array(np.shape(self.b[i]), dtype=np.int32).tostring()
            s += self.b[i].tostring(order='C')
        return s

    def db_values(self, with_transfer=False):
        if with_transfer:
            assert len(self.transfer_names) in [
                2, 3], 'length of transfer names is not 2 or 3'
            if len(self.transfer_names) == 2:
                trhelp_names = [self.transfer_names[0],
                                'none', self.transfer_names[1]]
            else:
                trhelp_names = self.transfer_names
            return self.hash_id, self.my_hash(), self.index, self.model, self.nin, self.nlayer, \
                   self.nhid1, trhelp_names[0], self.nhid2, trhelp_names[1], self.nout, trhelp_names[2], self.tostring()
        else:
            return self.hash_id, self.my_hash(), self.index, self.model, self.nin, \
                   self.nlayer, self.nhid1, self.nhid2, self.nout, self.tostring()

    @staticmethod
    def db_string(with_transfer=False):
        if with_transfer:
            return ' replica_hash, ann_hash, seq, model_id, nin, nlayer, '\
                   'nhid1, nhid1_transfer, nhid2, nhid2_transfer, nout, nout_transfer, ann_bin '
        else:
            return ' replica_hash, ann_hash, seq, model_id, nin, nlayer, nhid1, nhid2, nout, ann_bin '

    def predict(self, x):
        tmp = np.copy(x)
        dot = np.dot
        #print("inp ",tmp)
        for i in range(self.nlayer):
            # print("dot",i)
            # print(self.w[i].shape)
            # print(self.w[i])
            tmp = dot(self.w[i], tmp)
            #print("mul ",tmp)
            # print(tmp[:,:5])
            # print("plus",i)
            # print(self.b[i])
            tmp = tmp+self.b[i]
            #print("add ",tmp)
            # print(tmp[:,:5])
            # print("tranfer",i)
            tmp = self.transfer_funcs[i](tmp)
            #print("tfunc ",tmp)

            # print(tmp[:,:5])
        return tmp

    def tansig(self, x):
        return - 1.0 + 2.0 / (1.0 + np.exp(-2.0 * x))

    def logsig(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def purelin(self, x):
        return np.copy(x)

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    @staticmethod
    def wts_generator(name):
        """
        Should only be used to read the raw binary files generated by ML
        This is because of teh different ordering (col varies fasted, not row)
        """
        f = open(name, 'r')
        assert f, 'file not found'
        while f.tell() < os.fstat(f.fileno()).st_size:
            yield ANN.from_stream(f, import_bin=True)
        f.close()

    # property?
    def my_hash(self):
        return hashlib.sha1(self.tostring()).hexdigest() # returns 40, not 32


class REPLICA(object):

    @property
    def hash_id(self): 
        return self._hash_id

    @hash_id.setter
    def hash_id(self, value): 
        self._hash_id = value

    @property
    def pat(self): 
        return self._pat

    @pat.setter
    def pat(self, pat): 
        self._pat = pat

    @property
    def ncnv(self):
        nv = np.count_nonzero(np.array(self.pat == 0, dtype=np.int))
        nc = len(self.pat)
        return nc, nv

    @property
    def name(self): 
        return self._name

    @name.setter
    def name(self, name): self._name = name

    def __init__(self):
        self.name = ""  # make sure this exists
        return

    def split_boot(self, line, oldrosetta=False):
        if oldrosetta:
            pat = np.array([int(i) for i in line.split()], dtype=np.int8)
            # OK we made an error in the MATLAB code.  I intended to use SHA-1 there, but it defaulted to MD5
            # SHA-1 returns 32 hex char, and sha-1 returns 40
            # since we are 'stuck' with 32 (new rosetta) for the replicas, we should also use md5 for old rosetta
            # the replica-hash appears in the binary blob, and it is troublesome to change it now.
            # the hashes are supposed to be identifiers only.
            hash_id = hashlib.md5(pat.tostring()).hexdigest()
            # print(len(hash_id))
        else:
            tmp = line.split()
            hash_id = tmp[0]
            assert len(hash_id) == 32, 'hash_id not of len 32'
            pat = np.array([int(i) for i in tmp[1:]], dtype=np.int8)
        return hash_id, pat

    @staticmethod
    def from_stream(s, oldrosetta=False):
        rep = REPLICA()
        rep.hash_id, rep.pat = rep.split_boot(s.strip(), oldrosetta)
        return rep

    @staticmethod
    def from_query(hash_id, s):
        rep = REPLICA()
        assert len(hash_id) == 32, 'hash_id not len(32)'
        rep.hash_id = hash_id
        rep.pat = np.fromstring(s, dtype=np.int8, count=len(s), sep='')
        return rep

    def tostring(self):
        return self.pat.tostring()

    def db_values(self):
        return self.hash_id, self.name, self.tostring()

    @staticmethod
    def db_string():
        return ' hash, name, replica '


class ANN_res(object):

    @property
    def hash_id(self): 
        return self._hash_id

    @hash_id.setter
    def hash_id(self, value): 
        self._hash_id = value

    @property
    def index(self): 
        return self._res[0]

    @index.setter  # we might need this because we could renumber the index
    def index(self, value): 
        self._res[0] = value

    @property
    def model_id(self): 
        return self._res[1]

    @property
    def nhid(self):
        return self._res[2]

    @property
    def nc(self): 
        return self._res[3]

    @property
    def nv(self): 
        return self._res[4]

    @property
    def vgc_rmse(self):
        return self._res[5]

    @property
    def vgc_me(self): 
        return self._res[6]

    @property
    def vgv_rmse(self): 
        return self._res[7]

    @property
    def vgv_me(self): 
        return self._res[8]

    @property
    def ksc_rmse(self): 
        return self._res[9]

    @property
    def ksc_me(self): 
        return self._res[10]

    @property
    def ksv_rmse(self): 
        return self._res[11]

    @property
    def ksv_me(self): 
        return self._res[12]

    @property
    def nfail(self): 
        return self._res[13]

    @property
    def res(self): 
        return self._res

    @res.setter
    def res(self, res): 
        self._res = res

    _sep = ""

    def __init__(self):
        return

    def parse_res(self, line):
        subs = (('nf', self._sep),
                ('ksv_me', self._sep), ('ksv_rmse', self._sep),
                ('ksc_me', self._sep), ('ksc_rmse', self._sep), 
                ('vgv_me', self._sep), ('vgv_rmse', self._sep),
                ('vgc_me', self._sep), ('vgc_rmse', self._sep),
                ('nv', self._sep), ('nc', self._sep),
                ('nhid', self._sep), ('model', self._sep), ('i', self._sep))
        return reduce(lambda a, kv: a.replace(*kv), subs, line)

    def split_res(self, line):

        def convert_float(val_string):
            try:
                res = float(val_string)
            except ValueError:
                res = 999.999
            return res

        tmp = self.parse_res(line)
        tmp = tmp.split()
        hash_id = tmp[0]
        assert len(hash_id) == 32, 'hash_id not of len 32'
        conv = (int, int, int, int, int, convert_float, convert_float, convert_float,
                convert_float, convert_float, convert_float, convert_float, convert_float, int)
        res = [conv[i](a) for (i, a) in enumerate(tmp[1:])]
        return hash_id, res

    def db_values(self):
        return self.hash_id, self.index, self.model_id, self.nhid, self.nc, self.nv, self.vgc_rmse, self.vgc_me, \
               self.vgv_rmse, self.vgv_me, self.ksc_rmse, self.ksc_me, self.ksv_rmse, self.ksv_me, self.nfail

    @staticmethod
    def db_string():
        return ' replica_hash, seq, model_id, nhid, nc, nv, vgc_rmse, vgc_me, '\
               'vgv_rmse, vgv_me, ksc_rmse, ksc_me, ksv_rmse, ksv_me, nfail '

    @staticmethod
    def from_stream(s):
        ann_res = ANN_res()
        ann_res.hash_id, ann_res.res = ann_res.split_res(s.strip())
        return ann_res

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class PS(object):

    @property
    def var_names(self): 
        return self._var_names

    def __init__(self):
        self._var_names = []

    def compute_scales(self, data, var_names, var_pos, ymin=-1.0, ymax=1.0):

        self.xmin = np.nanmin(data, axis=1)
        self.xmax = np.nanmax(data, axis=1)
        self.nvar = len(self.xmin)
        self.xmin = self.xmin.reshape((self.nvar, 1))
        self.xmax = self.xmax.reshape((self.nvar, 1))
        # print(self.xmin.shape)
        # print(self.xmax.shape)

        self.ymin = np.ones((self.nvar, 1), dtype=float)*ymin
        self.ymax = np.ones((self.nvar, 1), dtype=float)*ymax
        # print(self.ymin.shape)
        # print(self.ymax.shape)

        self.gain = (self.ymax-self.ymin)/(self.xmax-self.xmin)
        # print(self.gain)
        self.gain = self.gain.reshape((self.nvar, 1))  # convert to matrix
        # print(self.gain)
        # print(self.gain.shape)
        self.offset = self.xmin

        assert len(var_names) == self.nvar, 'len(var_names)!=self.nvar'
        assert len(var_names) == self.nvar, 'len(var_names)!=self.nvar'
        self._var_names = var_names
        self.var_pos = var_pos

    @staticmethod
    def from_data(data, var_names, var_pos, ymin=-1.0, ymax=1.0):
        PSdata = PS()
        PSdata.compute_scales(data, var_names, var_pos, ymin=-1.0, ymax=1.0)
        return PSdata

    @staticmethod
    def from_DB(cursor, model_id, model_var_table):
        assert cursor, 'bad cursor'
        PSdata = PS()
        sql_query =  'SELECT var_name, V.var_id, xmin, xmax, ymin, ymax, gain, offset, sco, scs, sct, data_min, data_max '\
                    f'FROM `minmax` as M JOIN `{model_var_table}` as V ON (M.var_id=V.var_id) WHERE V.model_id= {model_id}  ORDER by `var_pos` ;'
        cursor.execute(sql_query)
        PSdata.parse_query(cursor)  # has side effects
        return PSdata

    def parse_query(self, cursor):

        data = list(cursor)
        # print(data)
        self.nvar = len(data)

        xmin = np.zeros((self.nvar,), dtype=float)
        xmax = np.zeros((self.nvar,), dtype=float)
        ymin = np.zeros((self.nvar,), dtype=float)
        ymax = np.zeros((self.nvar,), dtype=float)
        gain = np.zeros((self.nvar,), dtype=float)
        offset = np.zeros((self.nvar,), dtype=float)

        # offset traditional unscaling of Rostta output
        sco = np.zeros((self.nvar,), dtype=float)
        # slope traditional unscaling of Rostta output
        scs = np.zeros((self.nvar,), dtype=float)
        # 0: no transform, 1: log10 transform
        sct = np.zeros((self.nvar,), dtype=np.int)

        d_min = np.zeros((self.nvar,), dtype=float)
        d_max = np.zeros((self.nvar,), dtype=float)

        # self._var_names=[] # moved to _init_
        for i, row in enumerate(data):
            # somewhow I cannot figure out how to this with property
            self._var_names.append(row[0].lower())
            xmin[i] = row[2]
            xmax[i] = row[3]
            ymin[i] = row[4]
            ymax[i] = row[5]
            gain[i] = row[6]
            offset[i] = row[7]

            sco[i] = row[8]
            scs[i] = row[9]
            sct[i] = row[10]

            d_min[i] = row[11]
            d_max[i] = row[12]

        # convert to pseudo 2D
        self.xmin = xmin[:, np.newaxis]
        self.xmax = xmax[:, np.newaxis]
        self.ymin = ymin[:, np.newaxis]
        self.ymax = ymax[:, np.newaxis]
        self.gain = gain[:, np.newaxis]
        self.offset = offset[:, np.newaxis]

        # some numpy trickery (convert from 1D to 3D):
        self.sco = sco[np.newaxis, :, np.newaxis]
        self.scs = scs[np.newaxis, :, np.newaxis]
        self.sct = sct[np.newaxis, :, np.newaxis]

        # pseudo 2D for data sanity checking (input needs to be in [min..max] (inclusive)
        self.data_min = d_min[:, np.newaxis]
        self.data_max = d_max[:, np.newaxis]

    def fwd_mapminmax(self, x):
        return (x - self.xmin) * self.gain + self.ymin

    def bwd_mapminmax(self, y):
        return (y - self.ymin) / self.gain + self.offset

    def unscale(self, res):
        return res * self.scs + self.sco

    def check_data(self, data_in):
        # yield 2D array of bools
        nvar, nsamp = np.shape(data_in)
        # print("MINIMUM")
        # print(self.data_min)
        # print("MAXIMUM")
        # print(self.data_max)
        res = np.logical_and(np.greater_equal(data_in, self.data_min),
                             np.less_equal(data_in, self.data_max))  # must be between these values
        # convert 1D array (sample basis)
        res_sample = np.all(res, axis=0)
        # check that sand+silt+clay exist in var_names
        ssc_set = {'sand', 'silt', 'clay'}
        vnset = set(self.var_names)
        # print(self.var_names)
        if ssc_set.issubset(vnset):
            # sand, silt,clay are input and must sum to [99..101]
            ssc_sum = np.zeros((nsamp,), dtype=float)
            for s in ssc_set:
                ssc_sum += data_in[self.var_names.index(s)]
            res_ssc = np.logical_and(np.greater_equal(ssc_sum, 99.0), np.less_equal(
                ssc_sum, 101.0))  # must be between these values
            res_sample = np.logical_and(res_sample, res_ssc)
        # so return a 1D array whether the input is valid (true) or not
        return res_sample

    def DB_store(self, cursor):
        assert cursor, 'bad cursor'
        sql_insert = 'INSERT INTO `minmax` ( var_name, var_pos, xmin, xmax, ymin, ymax, gain, offset ) '\
                     'VALUES (%s, %s, %s, %s, %s, %s, %s, %s )'
        vals = [(self.var_names[i], self.var_pos[i], self.xmin[i, 0], self.xmax[i, 0], self.ymin[i,
                 0], self.ymax[i, 0], self.gain[i, 0], self.offset[i, 0]) for i in range(self.nvar)]
        cursor.executemany(sql_insert, vals)

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class ANN_MODEL(object):

    @property
    def ann_sequence(self): 
        return self._ann_sequence

    @ann_sequence.setter
    def ann_sequence(self, value): 
        self._ann_sequence = value

    @property
    def nmodel(self): 
        return len(self.ann_sequence)

    @property
    # hack! ann_sequence[0] should exist
    def nout(self): 
        return self.ann_sequence[0].nout

    @property
    def output_var(self): 
        return self.PS_data_out.var_names

    @property
    def model_id(self): 
        return self._model_id

    @model_id.setter
    def model_id(self, value): 
        self._model_id = value

    @property
    def input_var(self): 
        return self.PS_data_in.var_names

    def __init__(self, model_id, db):

        # print("HERE",db.conn.text_factory)
        with closing(db.get_cursor()) as cursor:

            #print(self.ann_query_clause % (model_id))
            cursor.execute(f"SELECT A.replica_hash, A.seq, A.model_id, A.nin, A.nlayer, A.nhid1, A.nhid1_transfer, "
                           f"A.nhid2, A.nhid2_transfer, A.nout,A.nout_transfer, A.ann_bin "
                           f"FROM `Ann` as A JOIN Ann_res AS R ON (A.replica_hash=R.replica_hash) "
                           f"WHERE A.model_id= {model_id} and R.model_id=A.model_id and R.nfail=0")
            # this should be its own class, with PS for input and output, plus scaling
            tmp_list = list(cursor)
            # print(len(tmp_list))
            # print(tmp_list[0:4])

        # whoops, next line needs to be out of loop...
        self.ann_sequence = [ANN.from_stream(io.BytesIO(ann_bin), nlayer=nlayer, transfers=[nhid1_transfer, nhid2_transfer, nout_transfer]) for (
            hash_id, seq, model_id, nin, nlayer, nhid1, nhid1_transfer, nhid2, nhid2_transfer, nout, nout_transfer, ann_bin) in tmp_list]

        assert len(self.ann_sequence) > 0, "Error cannot get the ANNs from the DB"

        self.model_id = model_id

        with closing(db.get_cursor()) as cursor:
            self.PS_data_out = PS.from_DB(cursor, model_id, 'Models_out_var')

        with closing(db.get_cursor()) as cursor:
            self.PS_data_in = PS.from_DB(cursor, model_id, 'Models_in_var')

    def get_variables(self, model_id, db, table):
        sql_string = f'SELECT `model_id`,`var_id`,`var_pos` FROM `{table}`  '\
                     f'WHERE `model_id`= {self.model_id} ORDER BY `var_pos`;'
        with closing(db.get_cursor()) as cursor:
            cursor.execute(sql_string)
        variable_ids = [record[1] for record in list(cursor)]
        return variable_ids

    def predict(self, data):

        # print('data_in_un',data[:,:10])
        nvar, nsamp = np.shape(data)
        data_bool = self.PS_data_in.check_data(data)
        nsamp_valid = np.sum(data_bool)
        # selects only valid samples
        data_valid = np.compress(data_bool, data, axis=1)
        data_ind = np.nonzero(data_bool)
        # above could be decorator

        data_in_mm = self.PS_data_in.fwd_mapminmax(data_valid)
        res_tmp = np.zeros((self.nmodel, self.nout, nsamp_valid), dtype=float)

        for i, ann in enumerate(self.ann_sequence):
            try:
                res = ann.predict(data_in_mm)
            except FloatingPointError:
                print((i, self.model_id, ann.hash_id))
            res_tmp[i] = res

        # res_tmp=self.PS_data_out.bwd_mapminmax(res_tmp) #Yonggen
        res_tmp = self.PS_data_out.unscale(res_tmp)

        if nsamp == nsamp_valid:
            res_fin = res_tmp
        elif nsamp > nsamp_valid:
            res_fin = np.ones((self.nmodel, self.nout, nsamp),
                              dtype=float)*-9.9
            res_fin[:, :, data_ind[0]] = res_tmp
        else:
            raise Exception("ERROR nsamp_valid > nsamp (should be impossible")

        return res_fin, self.PS_data_out.var_names, data_bool


class PTF_Model(object):

    # model_no can be a compound model (more than one ann's, such as old rosetta)
    # model_id should be an individual model

    @property
    def model_no(self): 
        return self._model_no

    @model_no.setter
    def model_no(self, val): 
        self._model_no = val

    @property
    def model_name(self): 
        return self._model_name

    @model_name.setter
    def model_name(self, val): 
        self._model_name = val

    @property
    def model_id(self): 
        return self._model_id

    @model_id.setter
    def model_id(self, val): 
        self._model_id = val

    @property
    def model_seq(self): 
        return self._model_seq

    @model_seq.setter
    def model_seq(self, val): 
        self._model_seq = val

    @property
    def input_var(self): 
        return self.ann_models[0].PS_data_in.var_names  # the [0] is a hack

    def __init__(self, model_no, db):
        sql_string = f'SELECT `model_no`,`model_name`,`model_id`,`model_seq` FROM `Models_names` '\
                     f'WHERE `model_no`= {model_no} ORDER BY `model_seq` ;'
        self.model_no = model_no
        with closing(db.get_cursor()) as cursor:
            cursor.execute(sql_string)

            res = list(cursor)
            self.nmodel = len(res)
            assert self.nmodel > 0, 'nmodel<=0! in PTF_MODEL'
            assert self.nmodel < 3, 'nmodel is not sensible'

            self.model_name = res[0][1]
            self.model_id = []
            self.model_seq = []
            self.ann_models = []
            for i in range(self.nmodel):
                self.model_id.append(res[i][2])
                self.model_seq.append(res[i][3])
                self.ann_models.append(ANN_MODEL(self.model_id[i], db))

    def sum_stat(self, res, nvar, nsamp):
        def skewkurt(res, mean, std):

            adev = (res-mean)/std
            s = np.power(adev, 3).mean(axis=0)
            k = np.power(adev, 4).mean(axis=0)-3.0
            return s, k

        avg = np.mean(res, axis=0)
        std = np.zeros((nvar, nsamp), dtype=float)
        skew = np.zeros((nvar, nsamp), dtype=float)
        kurt = np.zeros((nvar, nsamp), dtype=float)
        # print(np.shape(res))
        cov = np.zeros((nvar, nvar, nsamp), dtype=float)
        for i in range(nsamp):
            # print(np.shape(res[:,:,i]))
            cov[:, :, i] = np.cov(res[:, :, i], rowvar=False, ddof=0)
            std[:, i] = np.sqrt(np.diag(cov[:, :, i]))
            s, k = skewkurt(res[:, :, i], avg[:, i], std[:, i])

            skew[:, i] = s
            kurt[:, i] = k

        return avg, std, cov, skew, kurt

    def predict(self, data_in, summary_data=True):

        # make sure data is a numpy thing
        data = np.array(data_in, dtype=float)

        # clumsy code....
        varout = []
        res = []
        data_bool = []
        nout_total = 0

        # need to be intelligent how data offered, transpose if needed? How to deal with square matrices?
        nin, nsamp = np.shape(data)
        # SELECT CODE HERE

        for i in range(self.nmodel):  # nmodel refers to the possibility of HYBRID models
            # TODO: for oldrosetta unsat_k model we need to put output in data
            #print("model_no %s, model_id %s" %(self.model_no,self.ann_models[i].model_id))
            # res (3D), varnames (1D), bool of valid data (1D?)
            r, v, b = self.ann_models[i].predict(data)
            # print(np.shape(r))
            res.append(r)
            varout.append(v)
            data_bool.append(b)
            # TODO: we can pull this out of underlying classes
            nout_total += len(v)

        if summary_data:
            # summarize in means and stdev, alpha-stable analysis indicates that
            # results are almost normal
            var_names = []

            sum_res_mean = np.zeros((nout_total, nsamp), dtype=float)
            sum_res_std = np.zeros((nout_total, nsamp), dtype=float)
            sum_res_skew = np.zeros((nout_total, nsamp), dtype=float)
            sum_res_kurt = np.zeros((nout_total, nsamp), dtype=float)
            sum_res_cov = np.zeros((nout_total, nout_total, nsamp), dtype=float)
            sum_res_bool = np.zeros((nout_total, nsamp), dtype=bool)
            offset = 0
            for i in range(self.nmodel):  # nmodel refers to the possibility of HYBRID models
                var_names.extend(varout[i])  # could indicate log units
                nvar = len(varout[i])
                mean, std, cov, skew, kurt = self.sum_stat(res[i], nvar, nsamp)

                sum_res_mean[offset:offset+nvar, :] = mean
                sum_res_std[offset:offset+nvar, :] = std
                sum_res_cov[offset:offset+nvar, offset:offset+nvar] = cov
                sum_res_skew[offset:offset+nvar, :] = skew
                sum_res_kurt[offset:offset+nvar, :] = kurt
                sum_res_bool[offset:offset+nvar, :] = data_bool[i]
                offset += nvar
            # remember that alpha, n,ks are log10
            res_dict = dict(var_names=[x.decode('utf-8') for x in var_names],
                            sum_res_mean=sum_res_mean,
                            sum_res_std=sum_res_std,
                            sum_res_cov=sum_res_cov,
                            sum_res_skew=sum_res_skew,
                            sum_res_kurt=sum_res_kurt,
                            sum_res_bool=sum_res_bool)
        else:
            res_dict = dict(varout=varout, res=res, data_bool=data_bool)

        res_dict['nsamp'] = nsamp
        res_dict['nout'] = nout_total
        res_dict['nin'] = nin

        return res_dict
