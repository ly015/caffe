#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void OneWayLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size()) << "The number of input and output must be same,";
}

template <typename Dtype>
void OneWayLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for(int i = 0; i < bottom.size(); i ++) {
    top[i]->ReshapeLike(*bottom[i]);
  }
}

template <typename Dtype>
void OneWayLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for(int i = 0; i < bottom.size(); i++) {
    caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), top[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void OneWayLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for(int i = 0; i < bottom.size(); i++) {
    if(propagate_down[i])
      caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(OneWayLayer);
#endif

INSTANTIATE_CLASS(OneWayLayer);
REGISTER_LAYER_CLASS(OneWay);

}  // namespace caffe
