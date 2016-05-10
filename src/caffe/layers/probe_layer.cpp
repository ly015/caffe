#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ProbeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size()) << "Input dose not match output!";
}

template <typename Dtype>
void ProbeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  for(int i = 0; i < bottom.size(); i++) {
    top[i]->ReshapeLike(*(bottom[i]));
  }
}

template <typename Dtype>
void ProbeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for(int i = 0; i < bottom.size(); i ++) {
    top[i]->ShareData(*bottom[i]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ProbeLayer);
#endif

INSTANTIATE_CLASS(ProbeLayer);
REGISTER_LAYER_CLASS(Probe);
}  // namespace caffe
