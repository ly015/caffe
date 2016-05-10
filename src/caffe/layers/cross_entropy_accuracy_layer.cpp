#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void CrossEntropyAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of prediction and number of label must be equal";
  
  num_label_ = bottom[0]->count() / bottom[0]->num();
  top[0]->Reshape(1, num_label_, 1, 1);
  if(top.size() > 1) {
    top[1]->Reshape(1, num_label_, 1, 1);
  }
  if(top.size() > 2) {
    top[2]->Reshape(1, num_label_, 1, 1);
  }
}

template <typename Dtype>
void CrossEntropyAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  vector<int> accuracy(num_label_, 0);
  vector<int> count(num_label_, 0);
  const Dtype* prediction = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  int num = bottom[0]->num();
  for(int i = 0; i < num; i++) {
    for(int j = 0; j < num_label_; j++) {
      if(has_ignore_label_ && static_cast<int>(label[i * num_label_ + j]) == ignore_label_)
        continue;
      if((prediction[i * num_label_ + j] < 0 && label[i * num_label_ + j] < 0.5) ||
        (prediction[i * num_label_ + j] > 0 && label[i * num_label_ + j] >= 0.5))
        accuracy[j] ++;
      count[j] ++;
    }
  }

  for(int i = 0; i < num_label_; i ++) {
    top[0]->mutable_cpu_data()[i] = Dtype(accuracy[i]) / count[i];
    if(top.size() > 1)
      top[1]->mutable_cpu_data()[i] = accuracy[i];
    if(top.size() > 2)
      top[2]->mutable_cpu_data()[i] = count[i];
  }
}


INSTANTIATE_CLASS(CrossEntropyAccuracyLayer);
REGISTER_LAYER_CLASS(CrossEntropyAccuracy);

}  // namespace caffe
