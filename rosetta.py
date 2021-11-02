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

import numpy as np
from time import time
from contextlib import closing

from ann import PTF_Model
from db import DB
from pprint import pprint

class Rosetta(object):
    def __init__(self, model_no, debug=True):
        """
        :param model_no: Specifies model (note model 1 and 101 are still missing: textural tables of parameters)
            2 NEW SSC (sand, silt, clay)
            3 NEW SSC BD (sand, silt, clay, bulk density)
            4 NEW SSC BD TH33 (sand, silt, clay, bulk density, theta at 33 kPa)
            5 NEW SSC BD TH33 TH1500 (sand, silt, clay, bulk density, theta at 33 kPa and 1500 kPa)
            102 OLD SSC
            103 OLD SSC BD
            104 OLD SSC BD TH33
            105 OLD SSC BD TH33 TH1500

            UNITS
            SSC in weight %
            BD in g/cm3
            TH33 and T1500 as cm3/cm3
        """
        self.debug = debug

        with DB() as db:
            if debug:
                print("Getting models from database")
                t0 = time()
            self.ptf_model = PTF_Model(model_no, db)

            if debug:
                print("Getting models from database, done (%s s)" % (time()-t0))

    def predict_raw(self, data: np.array, summary_data=True):
        """
        :param data:
            np.array with (nsamp, ninput) dimensions

        :param summary_data:
            with summary_data=False you get the raw output WITHOUT Summary statistics
            output log10 of VG-alpha,VG-n, and Ks

        :return:
        res_dict
            shape: (nboot,nout,nsamp)
            Warning: old rosetta (models 102..105) can ONLY provide retention parameters (not Ks)
            New models (2..5) provide retention+ks
            This is because the retention and ks models are synchronized in the new models whereas they were calibrated
            on different datasets in the old model
            nboot is 1000, output is log10 of VG-alpha,VG-n, and Ks
        """
        debug = self.debug
        ptf_model = self.ptf_model

        nsamp, ninput = data.shape

        if debug:
            print("Processing")
            t0 = time()

        res_dict = ptf_model.predict(data, summary_data=summary_data)

        if debug:
            print("Processing done (%s s)" % (time() - t0))
            pprint(res_dict)

        return res_dict

    def predict(self, data: np.array):
        """
        :param data:
            np.array with (nsamp, ninput) dimensions

        :return:
        theta_r [cm3/cm3]
        theta_s [cm3/cm3]
        alpha  [1/cm]
        n
        Ks in [cm/day]
        standard deviations apply to the log10 forms for alpha, n and KS NOT their their antilog forms
        """

        res_dict = self.predict_raw(data, summary_data=True)

        vgm_mean = res_dict['sum_res_mean']
        return dict(theta_r=vgm_mean[0],
                    theta_s=vgm_mean[1],
                    alpha=10**vgm_mean[2],
                    npar=10**vgm_mean[3],
                    ks=10**vgm_mean[4])

    def _get_rosetta_cal_data(self, input_var):
        """
        This is given as an example
        note that we use the input_var to get model specific data
        so this function returns properly formatted input data!

        # IN THE CORRECT ORDER!!!
        > data = rosetta._get_rosetta_cal_data(input_var)

        # USER must figure out how to get his/her own data
        # Hard coded data example
        # So data must be offered as a NUMPY matrix with the shape (ninput_var, nsamp)  (this was a Matlab-ann convention)
        # Here we create a (nsamp,ninput) matrix that we transpose immediately to  (ninput,nsamp)
        # Warning: here we implicitly assume that we're using model 2 or 102 because we provide only sand, silt and clay %

        > data = np.array([[90,5,5],[1,1,98],[1,2,97],[2,3,95]],dtype=float).transpose()
        > data = np.array([[90,5,5,1.4],[1,1,98,1.0],[2,5,93,1.2]],dtype=float).transpose()
        > data = data_in.transpose()
        """
        debug = self.debug
        ptf_model = self.ptf_model
        sql_string = "SELECT " + ",".join(input_var) + " FROM `Data`"

        if debug:
            print("Getting data from database")
            print(sql_string)
            t0 = time()

        with DB() as db:
            with closing(db.get_cursor()) as cursor:
                cursor.execute(sql_string)
                data = np.array(list(cursor))
                data = data.transpose()

        if debug:
            print("Getting data from database, done (%s s)" % (time() - t0))

        return data


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description='Rosetta 3 pedotransfer function interface example.')
    parser.add_argument('model_n',  type=int)

    parser.add_argument('--raw', action='store_true', help='get raw data')

    parser.add_argument('-i', '--input', action='store', help='input from file ')
    parser.add_argument('-o', '--output', action='store',
                        help='store predicted data')

    args = parser.parse_args()
    print(args.input)

    data_in = np.genfromtxt(args.input, delimiter='', dtype=float).transpose()
    print(data_in)

    rosetta = Rosetta(args.model_n)

    if args.raw:
        res_dict = rosetta.predict_raw(data_in)
        print(res_dict)

    else:
        res_dict = rosetta.predict(data_in)
        print(res_dict)

        if args.output:
            if 'tests/validation' in os.path.abspath(args.output):
                print('Error, cannot overwrite files in tests/validation')
            vgm_new = np.stack((res_dict['theta_r'],
                                res_dict['theta_s'],
                                res_dict['alpha'],
                                res_dict['npar'],
                                res_dict['ks']))
            vgm_new = vgm_new.transpose()
            # output estimation
            np.savetxt(args.output, vgm_new, delimiter=',', fmt='%f')


# python3 rosetta.py 2 -i tests/input/test_input_H2w.txt -o tests/output/test_output_H2w.txt
# python3 rosetta.py 3 -i tests/input/test_input_H3w.txt -o tests/output/test_output_H3w.txt
