FractalNet Generation
=====================

The ``fractalnet.py`` file will generate a line for each component in the network
suitable for CIFAR-100. The number of filters, columns and reductions can
easily be changed by editing the file. To review the output in color::

    python fractalnet.py --color

Each line in the output can be interpreted by crox_ (``pip install crox``),
using the ``global.crox`` file to specify the replacements::

    python fractalnet.py > train.crox.prototxt
    crox train.crox.prototoxt > train.prototxt

Instead of using crox, you can also simply use ``fractalnet.py`` as
inspiration.

You will need to edit the data loading part of ``global.crox`` to link it up to your
experimental set up.

.. _crox: https://github.com/gustavla/crox
