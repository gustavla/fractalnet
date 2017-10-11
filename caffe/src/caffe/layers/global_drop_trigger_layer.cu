#include <vector>

#include "caffe/layers/global_drop_trigger_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GlobalDropTriggerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Dtype global_drop_ratio = this->layer_param().global_drop_trigger_param().global_drop_ratio();
    if (caffe_rng_rand() <= global_drop_ratio * UINT_MAX) {
        caffe_gpu_set(top[0]->count(), Dtype(1), top[0]->mutable_gpu_data());
    } else {
        caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
    }
    return;
}

template <typename Dtype>
void GlobalDropTriggerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top)
{
    return;
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalDropTriggerLayer);

} // namespace caffe
