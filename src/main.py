#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial

from PIL import Image

from util import get_filenames
from dehaze import dehaze

SP_IDX = (0,)  # for testing parameters
SP_PARAMS = ({'tmin': 0.2, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'w': 15, 'r': 40})


def generate_results(src, dest, generator):
    print('processing', src + '...')
    im = Image.open(src)
    dark, bright, rawt, refinedt, rawrad, rerad = generator(im)
    dark.save(dest % 'dark')
    bright.save(dest % 'bright')
    #rawt.save(dest % 'rawt')
    #refinedt.save(dest % 'refinedt')
    #rawrad.save(dest % 'radiance-rawt')
    #rerad.save(dest % 'radiance-refinedt')
    print('saved', dest)


def main():
    filenames = get_filenames()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=int,
                        choices=range(len(filenames)), # choices にリストを渡すと，引数の選択肢をその中にあるもののみに限定でき，それ以外を指定するとエラーになります。
                        help="index for single input image")
    parser.add_argument("-t", "--tmin", type=float, default=0.2,
                        help="minimum transmission rate")
    parser.add_argument("-w", "--window", type=int, default=15,
                        help="window size of dark channel")
    parser.add_argument("-r", "--radius", type=int, default=40,
                        help="radius of guided filter")

    args = parser.parse_args()

    if args.input is not None:
        src, dest = filenames[args.input]
        dest = dest.replace("%s", "%s-%d-%d-%d" % ("%s", args.tmin * 100, args.window, args.radius))
        generate_results(src, dest, partial(dehaze, tmin=args.tmin, w=args.window, r=args.radius))
    else:
        for idx in SP_IDX:
            src, dest = filenames[idx]
            for param in SP_PARAMS:
                newdest = dest.replace("%s", "%s-%d-%d-%d" % ("%s", param['tmin'] * 100, param['w'], param['r']))
                generate_results(src, newdest, partial(dehaze, **param)) # partial(func, params) returns func result which parameters are fixed by params. 

        for src, dest in filenames:
            generate_results(src, dest, dehaze)

if __name__ == '__main__':
    main()
