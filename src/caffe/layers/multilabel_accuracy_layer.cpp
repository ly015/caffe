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
void MultilabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void MultilabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  // vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(1, inner_num_, 1, 1);// accuracy in one batch
  if(top.size() > 1) {
    top[1]->Reshape(1, inner_num_, 1, 1);// number of reight preidcted sampels in one batch
  }
  if(top.size() > 2) {
    top[2]->Reshape(1, inner_num_, 1, 1);// number of valid samples in one batch
  }

  // LOG(INFO) << "Size of int: " << sizeof(int);
  // LOG(INFO) << "Size of float: " << sizeof(float);
  // getchar();
}

template <typename Dtype>
void MultilabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<Dtype> accuracy(inner_num_, 0);
  vector<Dtype> count(inner_num_, 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  // FILE* f_pre = fopen("Log_prediction.txt", "w");
  for (int i = 0; i < outer_num_; ++i) {
    // fprintf(f_pre, "\n%03d===", i);
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      // fprintf(f_pre, "%d ", bottom_data_vector[0].second);
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          accuracy[j] ++;          
          break;
        }
      }
      count[j] ++;
    }
  }
  // fclose(f_pre);
  
  // LOG(INFO) << "Accuracy: " << accuracy;
  // top[0]->mutable_cpu_data()[0] = accuracy / count;
  for(int i = 0; i < inner_num_; i++)
  {
    top[0]->mutable_cpu_data()[i] = accuracy[i] / count[i];
    if(top.size() > 1) {
      top[1]->mutable_cpu_data()[i] = accuracy[i];
    }
    if(top.size() > 2) {
      top[2]->mutable_cpu_data()[i] = count[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
  // Output the Label for check
  // FILE* f_lab = fopen("Log_label.txt", "w");
  // for(int i = 0; i < outer_num_; i++) {
  //   fprintf(f_lab, "\n%03d===", i);
  //   for(int j = 0; j < inner_num_; j++) {
  //     fprintf(f_lab, "%d ", static_cast<int>(bottom_label[i * inner_num_ + j]));
  //   }
  // }

  // fclose(f_lab);
}

INSTANTIATE_CLASS(MultilabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultilabelAccuracy);

}  // namespace caffe
