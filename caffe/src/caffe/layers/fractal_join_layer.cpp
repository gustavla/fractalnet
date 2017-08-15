#include <numeric>
#include <vector>

#include "caffe/layers/fractal_join_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FractalJoinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    FractalJoinParameter param = this->layer_param().fractal_join_param();
    vector<int> shape(4);
    shape = bottom[bottom.size() - 1]->shape();
    int num = param.global_drop().num();
    int channels = param.global_drop().channels();
    int height = param.global_drop().height();
    int width = param.global_drop().width();
    if (shape[0] == num && shape[1] == channels && shape[2] == height && shape[3] == width) {
        bottom_size = bottom.size() - 1;
    } else {
        bottom_size = bottom.size();
    }
    sum_path_input_ = param.sum_path_input();
    drops_.clear();
    for (int i = 0; i < bottom_size; ++i) {
        drops_.push_back(true);
    }
    global_drops_.clear();
    GlobalDropParameter global_param = param.global_drop();
    std::copy(global_param.undrop_path_ratio().begin(), global_param.undrop_path_ratio().end(), std::back_inserter(global_drops_));
    local_drops_.clear();
    std::copy(param.drop_path_ratio().begin(), param.drop_path_ratio().end(), std::back_inserter(local_drops_));
    if (local_drops_.size() == 0) {
        for (int i = 0; i < bottom_size; ++i) {
            local_drops_.push_back(0.0);
        }
    } else if (local_drops_.size() == 1) {
        for (int i = 0; i < bottom_size - 1; ++i) {
            local_drops_.push_back(local_drops_[0]);
        }
    }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    for (int i = 1; i < bottom_size; ++i) {
        CHECK(bottom[i]->shape() == bottom[0]->shape());
    }
    top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    fill(drops_.begin(), drops_.end(), true);
    if (this->phase_ == TRAIN) {
        unsigned int drop_mark_ = 0;
        bool global_drop_ = false;
        if (bottom_size < bottom.size()) {
            global_drop_ = bottom[bottom_size]->IsGlobalDrop();
        }
        if (global_drop_) {
            if (global_drops_.size() <= 1) {
                drop_mark_ = caffe_rng_rand() % bottom_size;
            } else {
                float sum = std::accumulate(global_drops_.begin(), global_drops_.end(), 0.0) * caffe_rng_rand() / UINT_MAX;
                drop_mark_ = bottom_size;
                while (sum > 0.0 && drop_mark_ > 0) {
                    sum -= global_drops_.back();
                    global_drops_.pop_back();
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
    caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
    for (int i = 0; i < bottom_size; ++i) {
        if (!drops_[i]) {
            caffe_axpy(top[0]->count(), mult, bottom[i]->cpu_data(), top[0]->mutable_cpu_data());
        }
    }
}

template <typename Dtype>
void FractalJoinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Dtype mult = 1.0 / total_undrop_;
    if (sum_path_input_) {
        mult = 1.0;
    }
    for (int i = 0; i < bottom_size; ++i) {
        if (propagate_down[i]) {
            if (!drops_[i]) {
                caffe_cpu_scale(top[0]->count(), mult, top[0]->cpu_diff(), bottom[i]->mutable_cpu_diff());
            } else {
                caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(FractalJoinLayer);
#endif

INSTANTIATE_CLASS(FractalJoinLayer);
REGISTER_LAYER_CLASS(FractalJoin);

} // namespace caffe
