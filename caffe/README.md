FractalNet in Caffe
===================
 
Copy all files into your Caffe folder. Then, add the following to your ``src/caffe/proto/caffe.proto`` file in ``LayerParameter``:

    optional FractalJoinParameter fractal_join_param = 1234;
	optional GlobalDropTriggerParameter global_drop_trigger_param = 1235;

Set ``1234`` and ``1235`` to whatever you want that is not in conflict with another layer's parameters. Also add the following to the bottom ``caffe.proto``:
```
message GlobalDropParameter {
  repeated float undrop_path_ratio = 1;
  optional int32 num = 2 [default = 1];
  optional int32 channels = 3 [default = 2];
  optional int32 height = 4 [default = 1];
  optional int32 width = 5 [default = 2];
}

message FractalJoinParameter {
  repeated float drop_path_ratio = 1;
  optional GlobalDropParameter global_drop = 2;
  optional float global_drop_this = 3 [default = 0.0];
}

message GlobalDropTriggerParameter {
  optional float global_drop_ratio = 1;
  optional int32 num = 2 [default = 1];
  optional int32 channels = 3 [default = 2];
  optional int32 height = 4 [default = 1];
  optional int32 width = 5 [default = 2];
}
```
1. In `GlobalDropTriggerParameter`, `global_drop_ratio` is the probability of global drop. A `blob`'s shape also is specified, I use a specified-shape-`blob` as a messager to tell the `FractalJoin` to perform global drop or not, it also is the last `bottom` of `FractalJoin`, so this shape must be different from the path's.
2. In `GlobalDropParameter`, `undrop_path_ratio` is the probability of the path which will be undropped. A `blob`'s shape also is specified, it must be equel to the shape specified in `GlobalDropTriggerParameter` of the same fractal block.
3. In `FractalJoinParameter`, `drop_path_ratio` is the probability of the path which will be dropped, `global_drop` specify the parameters of global drop. A element-wise sum option is added. `global_drop_this`  is the probability of doing global-drop as a single `Join`.

Re-compile and you should now have access to the ``FractalJoin`` and  ``GlobalDropTrigger`` layer.

> Usage: see `generation`.
