#include <vector>

#include "caffe/layers/global_drop_trigger_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GlobalDropTriggerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    vector<int> shape(4);
    shape[0] = this->layer_param().global_drop_trigger_param().num();
    shape[1] = this->layer_param().global_drop_trigger_param().channels();
    shape[2] = this->layer_param().global_drop_trigger_param().height();
    shape[3] = this->layer_param().global_drop_trigger_param().width();
    top[0]->Reshape(shape);
}

template <typename Dtype>
void GlobalDropTriggerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Dtype global_drop_ratio = this->layer_param().global_drop_trigger_param().global_drop_ratio();
    if (caffe_rng_rand() <= global_drop_ratio * UINT_MAX) {
        caffe_set(top[0]->count(), Dtype(1), top[0]->mutable_cpu_data());
    } else {
        caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
    }
    return;
}

template <typename Dtype>
void GlobalDropTriggerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    return;
}

#ifdef CPU_ONLY
STUB_GPU(GlobalDropTriggerLayer);
#endif

INSTANTIATE_CLASS(GlobalDropTriggerLayer);
REGISTER_LAYER_CLASS(GlobalDropTrigger);

} // namespace caffe
