#argparse accept parameters
import argparse
#redefine function
from PIL import Image
from functools import partial
from util import get_filenames
from dehaze import dehaze

#SP_IDX = (5, 6, 8, 12) # for testing parameters
"""
SP_PARAMS = (
    {'tmin': 0.2, 'Amax': 170, 'w': 15, 'r': 40},
    {'tmin': 0.5, 'Amax': 190, 'w': 15, 'r': 40},
    {'tmin': 0.5, 'Amax': 220, 'w': 15, 'r': 40})
"""

def generate_results(src, dest, generator):
    print 'processing', src + '...'
    im = Image.open(src)
    rawrad = generator(im)[0]
    rawrad.save(dest % 'radiance-rawt')
    """
    dark, rawt, refinedt, rawrad, rerad = generator(im)
    dark.save(dest % 'dark')
    rawt.save(dest % 'rawt')
    refinedt.save(dest % 'refinedt')
    rawrad.save(dest % 'radiance-rawt')
    rerad.save(dest % 'radiance-refinedt')
    """
    print "saved", dest

def main():
    filenames = get_filenames()
    for src, dest in filenames:
        generate_results(src, dest, dehaze)

if __name__ == '__main__':
    main()