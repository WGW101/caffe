// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);

  drop_full_sample_ = this->layer_param_.dropout_param().drop_full_sample();
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  if(!drop_full_sample_) {
    rand_vec_.Reshape(bottom[0]->shape());
  } else {
    rand_vec_.Reshape(bottom[0]->shape(0), 1, 1, 1);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count(1);
  const int num = bottom[0]->shape(0);
  if (this->phase_ == TRAIN) {
    // Create random numbers
    if(!drop_full_sample_) {
      caffe_rng_bernoulli(count*num, 1. - threshold_, mask);
      for (int i = 0; i < count*num; ++i) {
        top_data[i] = bottom_data[i] * mask[i] * scale_;
      }
    } else {
      caffe_rng_bernoulli(num, 1. - threshold_, mask);
      for (int i = 0; i < num; ++i) {
        for(int j = 0; j < count; ++j) {
          top_data[j] = bottom_data[j] * mask[i] * scale_;
        }
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count(1);
      const int num = bottom[0]->shape(0);
      if(!drop_full_sample_) {
        for (int i = 0; i < count*num; ++i) {
          bottom_diff[i] = top_diff[i] * mask[i] * scale_;
        }
      } else {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < count; ++j) {
            bottom_diff[j] = top_diff[j] * mask[i] * scale_;
          }
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
