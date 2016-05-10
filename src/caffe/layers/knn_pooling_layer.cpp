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
void KNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    K_ = this->layer_param_.knn_pooling_param().k_neighbor_num();
    CHECK_GT(K_, 0) << "K of KNN should be greater than 0";
    show_debug_info_ = this->layer_param_.knn_pooling_param().show_debug_info();
    GetInfo(bottom);
    

}

template <typename Dtype>
void KNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GetInfo(bottom);
  vector<int> shape(2);
  shape[0] = num_;
  shape[1] = dim_;
  top[0]->Reshape(shape);
  pool_mask_.Reshape(shape);
}

template <typename Dtype>
void KNNPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if(show_debug_info_) printf("KNNPooling: Forward...");
    if(K_ == 1) {
      CHECK_EQ(bottom[0]->count(), top[0]->count());
      caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
    } else {
      switch(this->layer_param_.knn_pooling_param().pool_method()) {
      case KNNPoolingParameter_KNNPoolMethod_MAX:
          int idx_top, idx_bottom;
          if(this->layer_param_.knn_pooling_param().pool_order() == KNNPoolingParameter_KNNPoolOrder_SAMPLE_PRIORITY) {
            for(int i = 0; i < num_; i++) {
              for(int j = 0; j < dim_; j++) {
                idx_top = i * dim_ + j;
                idx_bottom = (i * K_) * dim_ + j; // SAMPLE_PRIORITY
                for(int i_knn = 1; i_knn < K_; i_knn++) {
                  if(bottom[0]->cpu_data()[(i * K_ + i_knn) * dim_ + j] > // SAMPLE_PRIORITY
                    bottom[0]->cpu_data()[idx_bottom]) {
                    idx_bottom = (i * K_ + i_knn) * dim_ + j;
                  }
                }
                top[0]->mutable_cpu_data()[idx_top] = bottom[0]->cpu_data()[idx_bottom];
                pool_mask_.mutable_cpu_data()[idx_top] = idx_bottom;
              }
            }
          } else {
            for(int i = 0; i < num_; i++) {
              for(int j = 0; j < dim_; j++) {
                idx_top = i * dim_ + j;
                idx_bottom = i * dim_ + j; // K_PRIORITY
                for(int i_knn = 1; i_knn < K_; i_knn++) {
                  if(bottom[0]->cpu_data()[(i_knn * num_ + i) * dim_ + j] > // K_PRIORITY
                    bottom[0]->cpu_data()[idx_bottom]) {
                    idx_bottom = (i * K_ + i_knn) * dim_ + j;
                  }
                }
                top[0]->mutable_cpu_data()[idx_top] = bottom[0]->cpu_data()[idx_bottom];
                pool_mask_.mutable_cpu_data()[idx_top] = idx_bottom;
              }        
            }
          }
          break;
      case KNNPoolingParameter_KNNPoolMethod_AVE:
          // LOG(INFO) << "AVE KNNPooling to be added!";
          if(this->layer_param_.knn_pooling_param().pool_order() == KNNPoolingParameter_KNNPoolOrder_SAMPLE_PRIORITY) {
            for(int i = 0; i < num_; i ++) {
              caffe_set(dim_, Dtype(0), top[0]->mutable_cpu_data() + i * dim_);
              for(int k = 0; k < K_; k ++) {
                caffe_axpy(dim_, Dtype(1), bottom[0]->cpu_data() + (i * K_ + k) * dim_,
                  top[0]->mutable_cpu_data() + i * dim_);
              }
            }
            caffe_scal(top[0]->count(), Dtype(1) / K_, top[0]->mutable_cpu_data());
          } else {
            for(int i = 0; i < num_; i ++) {
              caffe_set(dim_, Dtype(0), top[0]->mutable_cpu_data() + i * dim_);
              for(int k = 0; k < K_; k ++) {
                caffe_axpy(dim_, Dtype(1), bottom[0]->cpu_data() + (k * num_ + i) * dim_,
                  top[0]->mutable_cpu_data() + i * dim_);
              }
            }
            caffe_scal(top[0]->count(), Dtype(1) / K_, top[0]->mutable_cpu_data());
          }
          break;
      }
    }
    if(show_debug_info_) printf("Done!\n");
}

template <typename Dtype>
void KNNPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
    if(show_debug_info_) printf("KNNPooling: Backward...");
    if(K_ == 1) {
      CHECK_EQ(bottom[0]->count(), top[0]->count());
      caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    } else {
      switch(this->layer_param_.knn_pooling_param().pool_method()) {
      case KNNPoolingParameter_KNNPoolMethod_MAX:
          caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
          for(int i = 0; i < top[0]->count(); i ++) {
            bottom[0]->mutable_cpu_diff()[pool_mask_.cpu_data()[i]] = top[0]->cpu_diff()[i];
          }
          break;
      case KNNPoolingParameter_KNNPoolMethod_AVE:       
          if(this->layer_param_.knn_pooling_param().pool_order() == KNNPoolingParameter_KNNPoolOrder_SAMPLE_PRIORITY) {
            for(int i = 0; i < num_; i ++) {
              for(int k = 0; k < K_; k ++) {
                caffe_copy(dim_, top[0]->cpu_diff() + i * dim_,
                  bottom[0]->mutable_cpu_diff() + (i * K_ + k) * dim_);
              }
            }
            caffe_scal(bottom[0]->count(), Dtype(1) / K_, bottom[0]->mutable_cpu_diff());
          } else {
            for(int i = 0; i < num_; i ++) {
              for(int k = 0; k < K_; k ++) {
                caffe_copy(dim_, top[0]->cpu_diff() + i * dim_,
                  bottom[0]->mutable_cpu_diff() + (k * num_ + i) * dim_);
              }
            }
            caffe_scal(bottom[0]->count(), Dtype(1) / K_, bottom[0]->mutable_cpu_diff());
          }
          break;
      }
    }
    if(show_debug_info_) printf("Done!\n");

}

template <typename Dtype>
void KNNPoolingLayer<Dtype>::GetInfo(const vector<Blob<Dtype>*>& bottom) {
    CHECK(bottom[0]->num() % K_ == 0) << "The number of imput is not a multiple of k = " << K_;
    num_ = bottom[0]->num() / K_;
    dim_ = bottom[0]->channels();
    if(show_debug_info_) {
      printf("KNNPooling: GetInfo(num_=%d, dim_%d, K_=%d)\n", num_, dim_, K_);
    }
}

#ifdef CPU_ONLY
STUB_GPU(KNNPoolingLayer);
#endif

INSTANTIATE_CLASS(KNNPoolingLayer);
REGISTER_LAYER_CLASS(KNNPooling);
}  // namespace caffe
