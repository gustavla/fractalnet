#include <vector>

#include "caffe/layers/fractal_join_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FractalJoinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  total_drops_ = 0;
  if (this->phase_ == TRAIN) {
    for (int i = 0; i < bottom.size(); ++i) {
      bool drop = (caffe_rng_rand() < uint_thresholds_[i]);
      drops_[i] = drop;
      if (drop) {
        ++total_drops_;
      }
    }

    // Check that all are not drop. If so, undrop a random one.
    if (total_drops_ == bottom.size()) {
      int choice = caffe_rng_rand() % bottom.size();
      drops_[choice] = false;
      --total_drops_;
    }
  } else {
    // Do not drop any if mode is not training
    fill(drops_.begin(), drops_.end(), false);
  }

  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
  Dtype mult = 1.0 / (bottom.size() - total_drops_);

  for (int i = 0; i < bottom.size(); ++i) {
    if (!drops_[i]) {
      caffe_gpu_axpy(top[0]->count(), mult, bottom[i]->gpu_data(), top[0]->mutable_gpu_data());
    }
  }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype mult = 1.0 / (bottom.size() - total_drops_);
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      if (drops_[i]) {
        caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_gpu_diff());
      } else {
        caffe_gpu_scale(top[0]->count(), mult, top[0]->gpu_diff(), bottom[i]->mutable_gpu_diff());
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FractalJoinLayer);

}  // namespace caffe
