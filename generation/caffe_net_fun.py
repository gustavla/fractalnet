from caffe import layers as L, params as P


def conv_relu_bn(bottom, ks, n_out, phase, stride=1, pad=0):
    conv = L.Convolution(
        bottom,
        kernel_size=ks,
        stride=stride,
        num_output=n_out,
        pad=pad,
        # param=[dict(lr_mult=1, decay_mult=1),
        #        dict(lr_mult=2, decay_mult=0)],
        # bias_term=False,
        # weight_filler=dict(type='msra'))
        weight_filler=dict(type='xavier', std=0.01),
        bias_filler=dict(type='constant', value=0))
    batch_norm = L.BatchNorm(conv, in_place=True, use_global_stats=(not phase))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    # relu = L.PowReU(scale, in_place=True, powers=.6180339887498949)
    return relu


def conv_bn(bottom, ks, n_out, phase, stride=1, pad=0):
    conv = L.Convolution(
        bottom,
        kernel_size=ks,
        stride=stride,
        num_output=n_out,
        pad=pad,
        # param=[dict(lr_mult=1, decay_mult=1),
        #        dict(lr_mult=2, decay_mult=0)],
        # bias_term=False,
        # weight_filler=dict(type='msra'))
        weight_filler=dict(type='xavier', std=0.01),
        bias_filler=dict(type='constant', value=0))
    batch_norm = L.BatchNorm(conv, in_place=True, use_global_stats=(not phase))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale


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
        # param=[dict(lr_mult=1, decay_mult=1),
        #        dict(lr_mult=2, decay_mult=0)],
        # bias_term=False,
        # weight_filler=dict(type='msra'))
        weight_filler=dict(type='xavier', std=0.01),
        bias_filler=dict(type='constant', value=0))


def fc_bn_relu_dropout(bottom, n_out, phase, ratio=.5):
    fc = full_connect(bottom, n_out)
    batch_norm = L.BatchNorm(fc, in_place=True, use_global_stats=(not phase))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return L.Dropout(relu, dropout_ratio=ratio, in_place=True)


def eltwise_all_relu(bottom1,
                     bottom2,
                     bottom3=None,
                     bottom4=None,
                     bottom5=None,
                     bottom6=None):
    if bottom6 is not None:
        return L.ReLU(
            L.Eltwise(bottom1, bottom2, bottom3, bottom4, bottom5, bottom6),
            in_place=True)
    elif bottom5 is not None:
        return L.ReLU(
            L.Eltwise(bottom1, bottom2, bottom3, bottom4, bottom5),
            in_place=True)
    elif bottom4 is not None:
        return L.ReLU(
            L.Eltwise(bottom1, bottom2, bottom3, bottom4), in_place=True)
    elif bottom3 is not None:
        return L.ReLU(L.Eltwise(bottom1, bottom2, bottom3), in_place=True)
    else:
        return L.ReLU(L.Eltwise(bottom1, bottom2), in_place=True)


def fractal_drop_relu(ratio,
                      bottom1,
                      bottom2,
                      bottom3=None,
                      bottom4=None,
                      bottom5=None,
                      bottom6=None):
    if bottom6 is not None:
        return L.ReLU(
            L.FractalJoin(
                bottom1,
                bottom2,
                bottom3,
                bottom4,
                bottom5,
                bottom6,
                drop_path_ratio=ratio),
            in_place=True)
    elif bottom5 is not None:
        return L.ReLU(
            L.FractalJoin(
                bottom1,
                bottom2,
                bottom3,
                bottom4,
                bottom5,
                drop_path_ratio=ratio),
            in_place=True)
    elif bottom4 is not None:
        return L.ReLU(
            L.FractalJoin(
                bottom1, bottom2, bottom3, bottom4, drop_path_ratio=ratio),
            in_place=True)
    elif bottom3 is not None:
        return L.ReLU(
            L.FractalJoin(bottom1, bottom2, bottom3, drop_path_ratio=ratio),
            in_place=True)
    else:
        return L.ReLU(
            L.FractalJoin(bottom1, bottom2, drop_path_ratio=ratio),
            in_place=True)


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
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        # drop_path_ratio=ratio)
    if bottom6 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            bottom5,
            bottom6,
            fractal_join_param=dict(
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        # drop_path_ratio=ratio)
    elif bottom5 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            bottom5,
            fractal_join_param=dict(
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        # drop_path_ratio=ratio)
    elif bottom4 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            bottom4,
            fractal_join_param=dict(
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        # drop_path_ratio=ratio)
    elif bottom3 is not None:
        return L.FractalJoin(
            bottom1,
            bottom2,
            bottom3,
            fractal_join_param=dict(
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        # drop_path_ratio=ratio)
    else:
        return L.FractalJoin(
            bottom1,
            bottom2,
            fractal_join_param=dict(
                # sum_path_input=True,
                drop_path_ratio=ratio,
                global_drop=dict(undrop_path_ratio=ratio_r)))
        #    drop_path_ratio=ratio)