#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/fractal_join_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class FractalJoinLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FractalJoinLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_d_(new Blob<Dtype>(1, 2, 1, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    filler.Fill(this->blob_bottom_d_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_bottom_vec_.push_back(blob_bottom_d_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~FractalJoinLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_bottom_d_;
    delete blob_top_;
  }

  Blob<Dtype> *const blob_bottom_a_;
  Blob<Dtype> *const blob_bottom_b_;
  Blob<Dtype> *const blob_bottom_c_;
  Blob<Dtype> *const blob_bottom_d_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_CASE(FractalJoinLayerTest, TestDtypesAndDevices);

TYPED_TEST(FractalJoinLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(FractalJoinLayerTest, TestSetUpDropPath) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.1);
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(FractalJoinLayerTest, TestSetUpGlobalDropPath) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.1);
  GlobalDropParameter *global_drop_param =
      fractal_join_param->mutable_global_drop();
  global_drop_param->add_undrop_path_ratio(0.0);
  global_drop_param->add_undrop_path_ratio(0.5);
  global_drop_param->add_undrop_path_ratio(0.0);
  Dtype *data_d = this->blob_bottom_d_->mutable_cpu_data();
  caffe_set(this->blob_bottom_d_->count(), Dtype(1), data_d);
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(FractalJoinLayerTest, TestNoDrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.0);
  fractal_join_param->add_drop_path_ratio(0.0);
  fractal_join_param->add_drop_path_ratio(0.0);
  Dtype *data_d = this->blob_bottom_d_->mutable_cpu_data();
  caffe_set(this->blob_bottom_d_->count(), Dtype(0), data_d);
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype *data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype *in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype *in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype *in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], (in_data_a[i] + in_data_b[i] + in_data_c[i]) / 3.0,
                1e-4);
  }
}

TYPED_TEST(FractalJoinLayerTest, TestDrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.0);
  fractal_join_param->add_drop_path_ratio(1.0);
  fractal_join_param->add_drop_path_ratio(0.0);
  Dtype *data_d = this->blob_bottom_d_->mutable_cpu_data();
  caffe_set(this->blob_bottom_d_->count(), Dtype(0), data_d);
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype *data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype *in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype *in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], (in_data_a[i] + in_data_c[i]) / 2.0, 1e-4);
  }
}

TYPED_TEST(FractalJoinLayerTest, TestGlobalDrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.0);
  fractal_join_param->add_drop_path_ratio(0.0);
  fractal_join_param->add_drop_path_ratio(0.0);
  GlobalDropParameter *global_drop_param =
      fractal_join_param->mutable_global_drop();
  global_drop_param->add_undrop_path_ratio(1.0);
  global_drop_param->add_undrop_path_ratio(0.0);
  global_drop_param->add_undrop_path_ratio(0.0);
  Dtype *data_d = this->blob_bottom_d_->mutable_cpu_data();
  caffe_set(this->blob_bottom_d_->count(), Dtype(1), data_d);
  shared_ptr<FractalJoinLayer<Dtype> > layer(
      new FractalJoinLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype *data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype *in_data_a = this->blob_bottom_a_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * 1.0, 1e-4);
  }
}

TYPED_TEST(FractalJoinLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(FractalJoinLayerTest, TestDropPathGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.15);
  fractal_join_param->add_drop_path_ratio(0.30);
  fractal_join_param->add_drop_path_ratio(0.45);
  FractalJoinLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

TYPED_TEST(FractalJoinLayerTest, TestGlobalDropPathGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FractalJoinParameter *fractal_join_param =
      layer_param.mutable_fractal_join_param();
  fractal_join_param->add_drop_path_ratio(0.15);
  fractal_join_param->add_drop_path_ratio(0.30);
  fractal_join_param->add_drop_path_ratio(0.45);
  GlobalDropParameter *global_drop_param =
      fractal_join_param->mutable_global_drop();
  global_drop_param->add_undrop_path_ratio(1.0);
  global_drop_param->add_undrop_path_ratio(0.0);
  global_drop_param->add_undrop_path_ratio(0.0);
  Dtype *data_d = this->blob_bottom_d_->mutable_cpu_data();
  caffe_set(this->blob_bottom_d_->count(), Dtype(1), data_d);
  FractalJoinLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

}  // namespace caffe

