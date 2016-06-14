#include <vector>

#include "caffe/layers/fractal_join_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FractalJoinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  FractalJoinParameter param = this->layer_param().fractal_join_param();

  thresholds_.clear();
  std::copy(param.drop_path_ratio().begin(),
            param.drop_path_ratio().end(),
            std::back_inserter(thresholds_));
  if (thresholds_.size() == 0) {
    // If none, fill with zeros (no drop-path)
    for (int i = 0; i < bottom.size(); ++i) {
      thresholds_.push_back(0.0);
    }
  } else if (thresholds_.size() == 1) {
    // If one value, replicate for each operand
    for (int i = 0; i < bottom.size() - 1; ++i) {
      thresholds_.push_back(thresholds_[0]);
    }
  }

  CHECK_EQ(thresholds_.size(), bottom.size());

  uint_thresholds_.clear();
  drops_.clear();
  for (int i = 0; i < thresholds_.size(); ++i) {
    DCHECK(thresholds_[i] >= 0.);
    DCHECK(thresholds_[i] <= 1.);
    uint_thresholds_.push_back(static_cast<unsigned int>(UINT_MAX * thresholds_[i]));
    drops_.push_back(false);
  }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype mult = 1.0 / (bottom.size() - total_drops_);

  for (int i = 0; i < bottom.size(); ++i) {
    if (!drops_[i]) {
      caffe_axpy(top[0]->count(), mult, bottom[i]->cpu_data(), top[0]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype mult = 1.0 / (bottom.size() - total_drops_);
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      if (drops_[i]) {
        caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
      } else {
        caffe_cpu_scale(top[0]->count(), mult, top[0]->cpu_diff(), bottom[i]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FractalJoinLayer);
#endif

INSTANTIATE_CLASS(FractalJoinLayer);
REGISTER_LAYER_CLASS(FractalJoin);

}  // namespace caffe
