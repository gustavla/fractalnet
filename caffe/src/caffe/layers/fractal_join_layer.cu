#include <numeric>
#include <vector>

#include "caffe/layers/fractal_join_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FractalJoinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    fill(drops_.begin(), drops_.end(), true);
    if (this->phase_ == TRAIN) {
        unsigned int drop_mark_ = 0;
        Dtype global_drop_this = this->layer_param().fractal_join_param().global_drop_this();
        bool global_drop_ = caffe_rng_rand() < (global_drop_this * UINT_MAX);
        if (bottom_size < bottom.size()) {
            const Dtype* data = bottom[bottom_size]->gpu_data();
            Dtype sum;
            caffe_gpu_asum(1, data, &sum);
            global_drop_ = (sum > Dtype(.5));
        }
        if (global_drop_) {
            if (global_drops_.size() != bottom_size) {
                drop_mark_ = caffe_rng_rand() % bottom_size;
            } else {
                Dtype sum = std::accumulate(global_drops_.begin(), global_drops_.end(), Dtype(0)) * caffe_rng_rand() / UINT_MAX;
                drop_mark_ = bottom_size;
                while (sum > Dtype(0) && drop_mark_ > 0) {
                    sum -= global_drops_[drop_mark_ - 1];
                    --drop_mark_;
                }
            }
            drops_[drop_mark_] = false;
            total_undrop_ = 1;
        } else {
            for (int i = 0; i < bottom_size; ++i) {
                bool drop = (caffe_rng_rand() < (local_drops_[i] * UINT_MAX));
                drops_[i] = drop;
                if (drop) {
                    ++drop_mark_;
                }
            }
            if (drop_mark_ == bottom_size) {
                drops_[caffe_rng_rand() % bottom_size] = false;
                --drop_mark_;
            }
            total_undrop_ = bottom_size - drop_mark_;
        }
    } else {
        fill(drops_.begin(), drops_.end(), false);
        total_undrop_ = bottom_size;
    }
    Dtype mult = 1.0 / total_undrop_;
    if (sum_path_input_) {
        mult = 1.0;
    }
    caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
    for (int i = 0; i < bottom_size; ++i) {
        if (!drops_[i]) {
            caffe_gpu_axpy(top[0]->count(), mult, bottom[i]->gpu_data(), top[0]->mutable_gpu_data());
        }
    }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Dtype mult = 1.0 / total_undrop_;
    if (sum_path_input_) {
        mult = 1.0;
    }
    for (int i = 0; i < bottom_size; ++i) {
        if (propagate_down[i]) {
            if (!drops_[i]) {
                caffe_gpu_scale(top[0]->count(), mult, top[0]->gpu_diff(), bottom[i]->mutable_gpu_diff());
            } else {
                caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_gpu_diff());
            }
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(FractalJoinLayer);

} // namespace caffe
