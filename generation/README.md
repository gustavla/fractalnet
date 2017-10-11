FractalNet Generation
=====================
# OLD
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
# NEW
The `caffe_net_fun.py` file includes some usefull function used for generating net in pycaffe.

The `fra_gl.py` file can generates full fractalnet.

The `fra_gl_train.prototxt` file, and `fra_gl_test.prototxt` file is the prototxt file for caffe.

The `fra_gl_test.jpg` image is visualized fractalnet.

