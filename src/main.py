#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, cv2
from functools import partial
from util import get_filenames
from dehaze import dehaze

SP_IDX = (0,)  # for testing parameters
SP_PARAMS = ({'tmin': 0.1, 'w': 15},)
             #{'tmin': 0.2, 'w': 15})

def generate_results(src, dest, generator):
    print('processing', src + '...')
    im = cv2.imread(src)
    #dark, bright, rawt, refinedt, rawrad, rerad = generator(im)
    generator(im)
    #dark.save(dest % 'dark')
    #bright.save(dest % 'bright')
    #rawt.save(dest % 'rawt')
    #refinedt.save(dest % 'refinedt')
    #rawrad.save(dest % 'radiance-rawt')
    #rerad.save(dest % 'radiance-refinedt')
    #print('saved', dest)

def main():
    filenames = get_filenames()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=int,
                        choices=range(len(filenames)), # choices にリストを渡すと，引数の選択肢をその中にあるもののみに限定でき，それ以外を指定するとエラーになります。
                        help="index for single input image")
    parser.add_argument("-t", "--tmin", type=float, default=0.1,
                        help="minimum transmission rate")
    parser.add_argument("-w", "--window", type=int, default=15,
                        help="window size of dark channel")

    args = parser.parse_args()

    if args.input is not None:
        src, dest = filenames[args.input]
        dest = dest.replace("%s", "%s-%d-%d" % ("%s", args.tmin * 100, args.window))
        generate_results(src, dest, partial(dehaze, tmin=args.tmin, w=args.window))
    else:
        for idx in SP_IDX:
            src, dest = filenames[idx]
            for param in SP_PARAMS:
                newdest = dest.replace("%s", "%s-%d-%d" % ("%s", param['tmin'] * 100, param['w']))
                generate_results(src, newdest, partial(dehaze, **param)) # partial(func, params) returns func result which parameters are fixed by params. 

        for src, dest in filenames:
            generate_results(src, dest, dehaze)

if __name__ == '__main__':
    main()
