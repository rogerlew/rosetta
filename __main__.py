
import os
import argparse
import numpy as np
import rosetta

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

_rosetta = rosetta.Rosetta(args.model_n)

if args.raw:
    res_dict = rosetta.predict_raw(data_in)
    print(res_dict)

else:
    res_dict = _rosetta.predict(data_in)
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