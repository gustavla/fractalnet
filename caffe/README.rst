FractalNet in Caffe
===================

Copy all files into your Caffe folder. Then, add the following to your ``src/caffe/proto/caffe.proto`` file in ``LayerParameter``::

    optional FractalJoinParameter fractal_join_param = 1234;

Set ``1234`` to whatever you want that is not in conflict with another layer's parameters. Also add the following to the bottom ``caffe.proto``::

   message FractalJoinParameter {
     repeated float drop_path_ratio = 1;
   }

Re-compile and you should now have access to the ``FractalJoin`` layer.

Usage
-----
Here is an example of how to join two layers, with 15% and 30% drop-path ratio, respectively::

    layer {
      type: "FractalJoin"
      bottom: "input1"
      bottom: "input2"
      top: "output"
      fractal_join_param {
        drop_path_ratio: 0.15
        drop_path_ratio: 0.30
      }
    }
