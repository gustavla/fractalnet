from __future__ import division, print_function, absolute_import
from string import Template
import textwrap
import sys

# SETTINGS
INPUT_SIZE = 32
REDUCTIONS = 5
COLUMNS = 3
CLASSES = 10
GLOBAL_DROP_PATH = False

FILTERS = [64, 128, 256, 512, 512]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4]
# Set drop-path in globals.crox


COLORS = dict(black='0;30', darkgray='1;30', red='1;31', green='1;32',
              brown='0;33', yellow='1;33', blue='1;34', purple='1;35', cyan='1;36',
              white='1;37', reset='0')

DEFAULT_COLORIZE = len(sys.argv) == 2 and sys.argv[1] == '--color'


def paint(s, color, colorize=DEFAULT_COLORIZE):
    if colorize:
        if color in COLORS:
            return '\033[{}m{}\033[0m'.format(COLORS[color], s)
        else:
            raise ValueError('Invalid color')
    else:
        return s


def conv(depth, red, num, dropout=0, bottom=None, separate=False):
    top = 'conv{}_{}'.format(depth, red)
    d = {}
    d['name'] = top
    if separate:
        top = 's_' + top
    d['top'] = top
    d['bottom'] = bottom
    d['num'] = num
    d['size'] = 3
    d['dropout'] = dropout
    d['pad'] = 1
    d['func'] = paint('conv', 'green')
    s = Template(":call $func $name $dropout $size $pad $num $bottom $top").substitute(d)
    return s, top


def pool(depth, red, bottom=None, separate=False):
    top = 'pool{}_{}'.format(depth, red)
    d = {}
    if separate:
        top = 's_' + top
    d['top'] = top
    d['bottom'] = bottom
    d['func'] = paint('pool2', 'purple')
    s = Template(":call $func $bottom $top").substitute(d)
    return s, top


def join(depth, red, bottoms=[]):
    top = bottoms[-1] + '_plus'
    d = {}
    d['top'] = top
    d['bottoms'] = '\n'.join('  bottom: {}'.format(b) for b in bottoms)
    d['bottoms_flat'] = ' '.join(bottoms)
    d['func'] = paint('join%d' % len(bottoms), 'yellow')
    s = Template(":call $func $bottoms_flat $top").substitute(d)
    return s, top


def inner(col, num, bottom=None, separate=False):
    top = 'prediction{}'.format(col)
    d = {}
    d['name'] = 'prediction0'
    if separate:
        top = 's_'+top
    d['top'] = top
    d['bottom'] = bottom
    d['num'] = num
    d['func'] = paint('inner', 'cyan')
    s = Template(":call $func $name $num $bottom $top").substitute(d)
    return s, top


def loss(col, weight, label, bottom=None, separate=False):
    if separate:
        top = 's_loss{}'.format(col)
    else:
        top = 'loss{}'.format(col)
    d = {}
    d['top'] = top
    d['bottom'] = bottom
    d['label'] = label
    d['weight'] = weight
    d['func'] = paint('loss', 'red')
    s = Template(":call $func $weight $bottom $label $top").substitute(d)
    return s, top

# Do one or two passes (two passes are used for tied global drop-path using separate columns)
PASSES = [False]
if GLOBAL_DROP_PATH:
    PASSES += [True]

replacements = {}

def get_rep(s):
    global replacements
    if s not in replacements:
        return s
    else:
        return get_rep(replacements[s])

def set_rep(c, r, value):
    global replacements
    replacements['{}_{}'.format(c, r)] = value

columns = [[] for _ in range(COLUMNS)]

print(':include globals.crox')
print(':call load-data data label')

print(paint('# Input size: {}'.format(INPUT_SIZE), 'darkgray'))

for separate in PASSES:
    lasts = ['data',] * COLUMNS
    ordinals = [0] * COLUMNS
    for r in range(REDUCTIONS):
        filters = FILTERS[r]
        dropout = DROPOUTS[r]

        L = 2**(COLUMNS - 1)
        for l in range(L):
            for c in reversed(range(COLUMNS)):
                minus_c = COLUMNS - c
                if (l + 1) % 2**(minus_c - 1) == 0:
                    s, lasts[c] = conv(c, ordinals[c], filters, dropout=dropout, bottom=lasts[c], separate=separate)
                    ordinals[c] += 1
                    print(s)

            if l == L - 1:
                for c in range(COLUMNS):
                    s, lasts[c] = pool(c, ordinals[c] - 1, bottom=lasts[c], separate=separate)
                    print(s)

            if not separate:
                for c in range(COLUMNS):
                    if l % 2 ** (2 + c) == 2 ** (1 + c) - 1:
                        num = 2 + c
                        s, last = join(c, ordinals[c], bottoms=lasts[-num:])
                        for i in range(COLUMNS-num, COLUMNS):
                            lasts[i] = last
                        print(s)
                        break

        print(paint('# Reduction: {}, spatial size: {}'.format(r+1, INPUT_SIZE // 2**(r+1)), 'darkgray'))

    if not separate:
        s, lasts[0] = inner(0, CLASSES, bottom=lasts[0], separate=separate)
        print(s)

        s, lasts[0] = loss(0, 1.0, 'label', bottom=lasts[0], separate=separate)
        print(s)

    else:
        for c in range(COLUMNS):
            s, lasts[c] = inner(c, CLASSES, bottom=lasts[c], separate=separate)
            print(s)

            s, lasts[c] = loss(c, 1/COLUMNS, 'label', bottom=lasts[c], separate=separate)
            print(s)
