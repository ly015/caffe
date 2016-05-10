#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ScoreWeightedAverageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ScoreWeightedAverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  num_score_ = bottom[0]->channels();
  num_input_ = bottom.size() - 1;
  CHECK_EQ(bottom[num_input_]->count(), num_score_ * num_input_) 
      << "Input weight size dose not match!";

  for(int i = 1; i < num_input_; i ++) {
    CHECK(bottom[0]->count() == bottom[i]->count() && bottom[0]->num() == bottom[i]->num())
        << "Input score size dose not match! " << i;
  }
  top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void ScoreWeightedAverageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_sample = bottom[0]->num();
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());

  for(int i_input = 0; i_input < num_input_; i_input ++) {
    for(int i_sample = 0; i_sample < num_sample; i_sample ++) {
      for(int j = 0; j < num_score_; j ++) {
        top[0]->mutable_cpu_data()[i_sample * num_score_ + j] += 
            bottom[i_input]->cpu_data()[i_sample * num_score_ + j] * 
            bottom[num_input_]->cpu_data()[i_input * num_score_ + j];
      }
    }
  }
}

template <typename Dtype>
void ScoreWeightedAverageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num_sample = bottom[0]->num();
  for(int i_input = 0; i_input < num_input_; i_input ++) {
    for(int i_sample = 0; i_sample < num_sample; i_sample ++) {
      for(int j = 0; j < num_score_; j ++) {
        bottom[i_input]->mutable_cpu_diff()[i_sample * num_score_ + j] =
            top[0]->cpu_diff()[i_sample * num_score_ + j] * 
            bottom[num_input_]->cpu_data()[i_input * num_score_ + j];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScoreWeightedAverageLayer);
#endif

INSTANTIATE_CLASS(ScoreWeightedAverageLayer);
REGISTER_LAYER_CLASS(ScoreWeightedAverage);

}  // namespace caffe
