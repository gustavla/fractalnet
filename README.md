FractalNet
==========

A fractal-based neural network architecture:

* `Project page <http://people.cs.uchicago.edu/~larsson/fractalnet/>`__
* `arXiv paper <https://arxiv.org/abs/1605.07648>`__

Drop-path
---------
We provide a reference implementation for the elementwise-mean layer with local
drop-path. There is still no public release of local+global, but we suggest
implementing this through tying weights. 

Caffe
~~~~~
See the ``caffe`` directory for code and more information.

Fractal pattern generation
--------------------------
Wiring up a fractal network manually would take hours, so we provide simple
Python scripts that will do it for you. See the ``generation`` directory for
code and more information.

Data augmentation
-----------------
We use a Python layer in Caffe to implement data augmentation. It is not yet
available here.

Cite
----
If this is useful to you, please consider citing us::

    @article{larsson2016fractalnet,
      title={FractalNet: Ultra-Deep Neural Networks without Residuals},
      author={Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory},
      journal={arXiv preprint arXiv:1605.07648},
      year={2016}
    }

