#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/global_drop_trigger_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class GlobalDropTriggerLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

protected:
    GlobalDropTriggerLayerTest()
        : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5))
        //        , blob_top_a_(new Blob<Dtype>())
        , blob_top_b_(new Blob<Dtype>())
    {
        // fill the values
        Caffe::set_random_seed(1701);
        FillerParameter filler_param;
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_);
        blob_bottom_vec_.push_back(blob_bottom_);
        //        blob_top_vec_.push_back(blob_top_a_);
        blob_top_vec_.push_back(blob_top_b_);
    }
    virtual ~GlobalDropTriggerLayerTest()
    {
        delete blob_bottom_;
        //        delete blob_top_a_;
        delete blob_top_b_;
    }
    Blob<Dtype>* const blob_bottom_;
    //    Blob<Dtype>* const blob_top_a_;
    Blob<Dtype>* const blob_top_b_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GlobalDropTriggerLayerTest, TestDtypesAndDevices);

TYPED_TEST(GlobalDropTriggerLayerTest, TestSetUp)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GlobalDropTriggerParameter* global_param = layer_param.mutable_global_drop_trigger_param();
    global_param->set_global_drop_ratio(0.0);
    shared_ptr<GlobalDropTriggerLayer<Dtype> > layer(
        new GlobalDropTriggerLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_b_->num(), 1);
    EXPECT_EQ(this->blob_top_b_->channels(), 2);
    EXPECT_EQ(this->blob_top_b_->height(), 1);
    EXPECT_EQ(this->blob_top_b_->width(), 2);
}

TYPED_TEST(GlobalDropTriggerLayerTest, TestPass)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GlobalDropTriggerParameter* global_param = layer_param.mutable_global_drop_trigger_param();
    global_param->set_global_drop_ratio(1.0);
    shared_ptr<GlobalDropTriggerLayer<Dtype> > layer(
        new GlobalDropTriggerLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_b_->cpu_data();
    const int count = this->blob_top_b_->count();
    //    const Dtype* in_data = this->blob_bottom_->cpu_data();
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(data[i], Dtype(0), 1e-4);
    }
    EXPECT_EQ(this->blob_top_b_->IsGlobalDrop(), true);
    EXPECT_EQ(this->blob_top_b_->num(), 1);
    EXPECT_EQ(this->blob_top_b_->channels(), 2);
    EXPECT_EQ(this->blob_top_b_->height(), 1);
    EXPECT_EQ(this->blob_top_b_->width(), 2);
}

TYPED_TEST(GlobalDropTriggerLayerTest, TestPassGlobal)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GlobalDropTriggerParameter* global_param = layer_param.mutable_global_drop_trigger_param();
    global_param->set_global_drop_ratio(0.0);
    shared_ptr<GlobalDropTriggerLayer<Dtype> > layer(
        new GlobalDropTriggerLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_b_->cpu_data();
    const int count = this->blob_top_b_->count();
    //    const Dtype* in_data = this->blob_bottom_->cpu_data();
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(data[i], Dtype(0), 1e-4);
    }
    EXPECT_EQ(this->blob_top_b_->IsGlobalDrop(), false);
    EXPECT_EQ(this->blob_top_b_->num(), 1);
    EXPECT_EQ(this->blob_top_b_->channels(), 2);
    EXPECT_EQ(this->blob_top_b_->height(), 1);
    EXPECT_EQ(this->blob_top_b_->width(), 2);
}
}
