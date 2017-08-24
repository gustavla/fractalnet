from __future__ import print_function
from caffe import layers as L, params as P, to_proto

import caffe_net_fun


def fractal_unit(bottom, n_out, phase):
    conv = bottom
    conv = caffe_net_fun.conv_relu_bn(
        conv, ks=3, n_out=n_out, phase=phase, pad=1)
    return conv


def fractal_drop_all(bottom, ratio, pool=False):
    ```
	if is the last FractalJoin, reductions will be taked
	```
    if pool:
        for __i in range(len(bottom) - 1):
            bottom[__i] = caffe_net_fun.max_pool(bottom[__i], ks=2, stride=2)
    return caffe_net_fun.fractal_drop(ratio, bottom)


def fractal_block(bottom, n_out, phase, total_level):
    ```
	n_out is the number of filters
	total_level is the columns
	```
    trigger = L.GlobalDropTrigger(bottom, global_drop_ratio=0.5)

    def recursive_struct(bt, level):
        ratio = 0.15
        if level <= 1:
            return [fractal_unit(bt, n_out, phase)], [ratio]
        else:
            unit, unit_ratio = recursive_struct(bt, level - 1)
            if len(unit) >= 2:
                unit = fractal_drop_all(unit + [trigger], unit_ratio)
            else:
                unit = unit[0]
            unit, unit_ratio = recursive_struct(unit, level - 1)
            res, res_ratio = recursive_struct(bt, 0)
            return unit + res, unit_ratio + res_ratio

    top, ratio = recursive_struct(bottom, total_level)
    return L.Dropout(
        fractal_drop_all(top + [trigger], ratio, True),
        dropout_ratio=0.5,
        in_place=True)


def caffe_net(lmdb, mean_file, batch_size=24, phase=False):
    data, label = L.Data(
        source=lmdb,
        backend=P.Data.LMDB,
        batch_size=batch_size,
        ntop=2,
        transform_param=dict(crop_size=32, mean_file=mean_file, mirror=phase))
    fractal_unit = fractal_block(data, 64, phase, 4)
    fractal_unit = fractal_block(fractal_unit, 128, phase, 4)
    fractal_unit = fractal_block(fractal_unit, 256, phase, 4)
    fractal_unit = fractal_block(fractal_unit, 512, phase, 4)
    fractal_unit = fractal_block(fractal_unit, 512, phase, 4)
    fc = caffe_net_fun.full_connect(fractal_unit, 10)
    loss = L.SoftmaxWithLoss(fc, label)
    if not phase:
        acc = L.Accuracy(fc, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def make_net(train_file, test_file, mean_file, train, test):
    with open(train_file, 'w') as f:
        print('name: "fra_gl"', file=f)
        print(caffe_net(train, mean_file, batch_size=64, phase=True), file=f)
    with open(test_file, 'w') as f:
        print('name: "fra_gl"', file=f)
        print(caffe_net(test, mean_file, batch_size=50), file=f)


if __name__ == '__main__':
    root_dir = '/path/to/your/cifar10'
    make_net('/path/to/your/want/to/save/fra_gl_train.prototxt',
             '/path/to/your/want/to/save/fra_gl_test.prototxt',
             root_dir + '/mean.binaryproto', root_dir + '/train',
             root_dir + '/test')
