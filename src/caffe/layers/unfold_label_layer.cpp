#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void UnfoldLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_label = this->layer_param_.unfold_label_param().num_label();

}

template <typename Dtype>
void UnfoldLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(2);
	top_shape[0] = bottom[0]->num();
	top_shape[1] = num_label;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void UnfoldLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int label_value;
    CHECK_EQ(bottom[0]->count() * num_label, top[0]->count());

    caffe_set(top[0]->count(), Dtype(0), top_data);	
    for(int i = 0; i < bottom[0]->count(); i ++) {
    	label_value = static_cast<int>(bottom_data[i]);
    	CHECK_GE(label_value, 0);
    	CHECK_GE(num_label, label_value);
    	top_data[i * num_label + label_value] = 1;
    }
}

template <typename Dtype>
void UnfoldLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
}

#ifdef CPU_ONLY
STUB_GPU(UnfoldLabelLayer);
#endif

INSTANTIATE_CLASS(UnfoldLabelLayer);
REGISTER_LAYER_CLASS(UnfoldLabel);

}  // namespace caffe
