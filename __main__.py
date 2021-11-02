
import os
import argparse
import numpy as np
import rosetta

parser = argparse.ArgumentParser(
    description='Rosetta 3 pedotransfer function interface example.')
parser.add_argument('model_n',  type=int)

parser.add_argument('--raw', action='store_true', help='get raw data')
parser.add_argument('--calc_wilting_point', action='store_true', help='calculate wilting point')
parser.add_argument('--calc_field_capacity', action='store_true', help='calculate field capacity')

parser.add_argument('-i', '--input', action='store', help='input from file ')
parser.add_argument('-o', '--output', action='store',
                    help='store predicted data')

args = parser.parse_args()
print(args.input)

data_in = np.genfromtxt(args.input, delimiter='', dtype=float)
print(data_in)

_rosetta = rosetta.Rosetta(args.model_n)

if args.raw:
    res_dict = rosetta.predict_raw(data_in)
    print(res_dict)

else:
    res_dict = _rosetta.predict(data_in,
                                calc_wilting_point=args.calc_wilting_point,
                                calc_field_capacity=args.calc_field_capacity)
    print(res_dict)

    if args.output:
        if 'tests/validation' in os.path.abspath(args.output):
            print('Error, cannot overwrite files in tests/validation')

        vgm_new = []
        for k, v in res_dict.items():
            vgm_new.append(v)

        vgm_new = np.stack(vgm_new).transpose()
        np.savetxt(args.output, vgm_new, delimiter=',', fmt='%f')


# python3 rosetta.py 2 -i tests/input/test_input_H2w.txt -o tests/output/test_output_H2w.txt
# python3 rosetta.py 3 -i tests/input/test_input_H3w.txt -o tests/output/test_output_H3w.txt