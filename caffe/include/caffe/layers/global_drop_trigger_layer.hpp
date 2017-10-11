#ifndef CAFFE_GLOBAL_DROP_TRIGGER_LAYER_HPP_
#define CAFFE_GLOBAL_DROP_TRIGGER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * \ingroup ttic
 * @brief a trigger which decides the fractal block will be global drop or not
 *
 * @author Liang Kang
 */
template <typename Dtype>
class GlobalDropTriggerLayer : public Layer<Dtype> {
public:
    explicit GlobalDropTriggerLayer(const LayerParameter& param)
        : Layer<Dtype>(param)
    {
    }
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "GlobalDropTrigger"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

} // namespace caffe

#endif // CAFFE_GLOBAL_DROP_TRIGGER_LAYER_HPP_
