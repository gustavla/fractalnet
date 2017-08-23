#ifndef CAFFE_FRACTAL_JOIN_LAYER_HPP_
#define CAFFE_FRACTAL_JOIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * \ingroup ttic
 * @brief Joins several blobs of the same size using a mean or sum operation.
 *        During training, operands can be stochastically dropped out to
 *        prevent co-adaptation of network paths.
 
 * @author Gustav Larsson
 * @author Liang Kang
 */
template <typename Dtype>
class FractalJoinLayer : public Layer<Dtype> {
public:
    explicit FractalJoinLayer(const LayerParameter& param)
        : Layer<Dtype>(param)
    {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "FractalJoin"; }
    virtual inline int MinBottomBlobs() const { return 2; }
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

    bool sum_path_input_;
    unsigned int total_undrop_;
    unsigned int bottom_size;
    std::vector<bool> drops_;
    std::vector<Dtype> global_drops_;
    std::vector<Dtype> local_drops_;
};

} // namespace caffe

#endif // CAFFE_FRACTAL_JOIN_LAYER_HPP_
