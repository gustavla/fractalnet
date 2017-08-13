from caffe import layers as L, params as P


def conv_relu_bn(bottom, ks, n_out, phase, stride=1, pad=0):
    conv = L.Convolution(
        bottom,
        kernel_size=ks,
        stride=stride,
        num_output=n_out,
        pad=pad,
        weight_filler=dict(type='xavier', std=0.01),
        bias_filler=dict(type='constant', value=0))
    batch_norm = L.BatchNorm(conv, in_place=True, use_global_stats=(not phase))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu


def max_pool(bottom, ks, stride=1, pad=0):
    return L.Pooling(
        bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=ks, stride=stride)


def mean_pool(bottom, ks, stride=1, pad=0):
    return L.Pooling(
        bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=ks, stride=stride)


def full_connect(bottom, n_out):
    return L.InnerProduct(
        bottom,
        num_output=n_out,
        weight_filler=dict(type='xavier', std=0.01),
        bias_filler=dict(type='constant', value=0))


def fractal_drop(ratio,
                 bottom1,
                 bottom2,
                 bottom3=None,
                 bottom4=None,
                 bottom5=None,
                 bottom6=None,
                 bottom7=None):
    ratio_r = ratio[:]
    ratio_r.reverse()
    if bottom7 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            bottom5,
            bottom6,
            bottom7,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
    if bottom6 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            bottom5,
            bottom6,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
    elif bottom5 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            bottom5,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
    elif bottom4 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
    elif bottom3 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
    else:
        return L.FractalJoin(
            bottom1,
            bottom2,
            fractal_join_param=dict(
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))