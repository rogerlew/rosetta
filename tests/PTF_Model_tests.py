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

import sys
import os

sys.path.insert(0, os.path.abspath('../../'))

from pprint import pprint

import numpy as np
from rosetta.db import DB
from rosetta.ann import PTF_Model

import unittest


class TestPTF_Model(unittest.TestCase):

    def test_102(self):
        # model 102 (ssc, old rosetta)
        with DB() as db:
            ptf_model = PTF_Model(102, db)

        data = np.array([[90, 5, 5], [40, 35, 25]], dtype=float).transpose()
        res_dict = ptf_model.predict(data, summary_data=True)

        self.assertEqual(['theta_r', 'theta_s', 'alpha', 'npar', 'ks'], res_dict['var_names'])
        sum_res_mean_val = np.array([[ 0.05153501,  0.07058601],
                                     [ 0.37694335,  0.41428159],
                                     [-1.47875897, -1.90366083],
                                     [ 0.39849954,  0.15826674],
                                     [ 2.50787295,  0.84132649]])

        np.testing.assert_allclose(res_dict['sum_res_mean'], sum_res_mean_val)

    def test_103(self):
        # model 102 (ssc, old rosetta)
        with DB() as db:
            ptf_model = PTF_Model(103, db)

        data = np.array([[90, 5, 5, 1.5], [40, 35, 25, 1.5]],
                        dtype=float).transpose()
        res_dict = ptf_model.predict(data, summary_data=True)

        self.assertEqual(['theta_r', 'theta_s', 'alpha', 'npar', 'ks'], res_dict['var_names'])
        sum_res_mean_val = np.array([[ 0.05158136,  0.06738687],
                                     [ 0.39064472,  0.3957376 ],
                                     [-1.47575018, -1.91920681],
                                     [ 0.41348101,  0.15979725],
                                     [ 2.55775783,  0.86256999]])

        np.testing.assert_allclose(res_dict['sum_res_mean'], sum_res_mean_val)

    def test_104(self):
        # model 102 (ssc, old rosetta)
        with DB() as db:
            ptf_model = PTF_Model(104, db)

        data = np.array([[90, 5, 5, 1.5, 0.1], [40, 35, 25, 1.5, 0.2]],
                        dtype=float).transpose()
        res_dict = ptf_model.predict(data, summary_data=True)


        self.assertEqual(['theta_r', 'theta_s', 'alpha', 'npar', 'ks'], res_dict['var_names'])
        sum_res_mean_val = np.array([[ 0.05394117,  0.06024026],
                                     [ 0.39119887,  0.37875298],
                                     [-1.44437367, -1.5459162 ],
                                     [ 0.3991555 ,  0.16001899],
                                     [ 2.55358626,  1.38956586]])

        np.testing.assert_allclose(res_dict['sum_res_mean'], sum_res_mean_val)

    def test_105(self):
        # model 102 (ssc, old rosetta)
        with DB() as db:
            ptf_model = PTF_Model(105, db)

        data = np.array([[90, 5, 5, 1.5, 0.1, 0.05], [
            40, 35, 25, 1.5, 0.2, 0.05]], dtype=float).transpose()
        res_dict = ptf_model.predict(data, summary_data=True)

        self.assertEqual(['theta_r', 'theta_s', 'alpha', 'npar', 'ks'], res_dict['var_names'])
        sum_res_mean_val = np.array([[ 0.03142677,  0.02666339],
                                     [ 0.39239906,  0.36292035],
                                     [-1.21869307, -1.95273874],
                                     [ 0.20618201,  0.15805979],
                                     [ 2.51549197,  1.34503962]])

        np.testing.assert_allclose(res_dict['sum_res_mean'], sum_res_mean_val)


if __name__ == '__main__':
    unittest.main()

