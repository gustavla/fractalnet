FractalNet
==========

A fractal-based neural network architecture:
[arXiv paper](https://arxiv.org/abs/1605.07648)

Drop-path
---------
I  provide a reference implementation for the elementwise layer with global and local drop-path.

Ideal
----
I use a new layer `GlobalDropTrigger` as a messager. It sends information to all the ` FractalJoin` in the same fractal block, and tells them drop path as global or local.

Caffe
-----
See the ``caffe`` directory for code and more information.

Fractal pattern generation
--------------------------
Wiring up a fractal network manually would take hours, so we provide simple Python scripts that will do it for you. See the ``generation`` directory for code and more information.

Data augmentation
-----------------
We use a Python layer in Caffe to implement data augmentation. It is not yet available here.

Cite
----
If this is useful to you, please consider citing us::

    @article{larsson2016fractalnet,
      title={FractalNet: Ultra-Deep Neural Networks without Residuals},
      author={Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory},
      journal={arXiv preprint arXiv:1605.07648},
      year={2016}
    }

