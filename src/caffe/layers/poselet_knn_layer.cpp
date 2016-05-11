// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		show_debug_info_ = this->layer_param_.poselet_knn_param().show_debug_info();
		threshold_ = this->layer_param_.poselet_knn_param().knn_threshold();
		K_ = this->layer_param_.poselet_knn_param().k_neighbor_num();
		num_ppp_ = this->layer_param_.poselet_knn_param().poselet_num_per_person();
		if_search_knn_ = (top.size() > 1)? true:false;
		poselet_select_mode_  = this->layer_param_.poselet_knn_param().poselet_select_mode();
		if(if_search_knn_)
			CHECK_GT(K_, 0);
	}

	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[3]->num() % num_ppp_, 0) << "Wrong PoseletInfo Number!";
		num_people_ = bottom[3]->num() / num_ppp_;
		num_scale_ = bottom[0]->num() / num_people_ / num_ppp_;
		dim_score_ = bottom[0]->count() / bottom[0]->num();
		dim_featureMap_ = bottom[1]->count() / bottom[1]->num();
		dim_feature_ = bottom[2]->count() / bottom[2]->num();
		num_psp_ = this->layer_param_.poselet_knn_param().num_select_poselet(); 
		if(num_psp_ == 0)
			num_psp_ = dim_score_;// select one poselet for each attribute
		CHECK_EQ(bottom[0]->num(), num_scale_ * num_people_ * num_ppp_) << "Wrong Score Number!";
		CHECK_EQ(bottom[1]->num(), num_scale_ * num_people_ * num_ppp_) << "Wrong FeatureMap Number!";
		CHECK_EQ(bottom[2]->num(), num_scale_ * num_people_ * num_ppp_) << "Wrong Feature Number";

		if(this->layer_param_.poselet_knn_param().output_mode() == PoseletKNNParameter_OutputMode_OUTPUT_SCORE) {
			// Output score
			top[0]->Reshape(num_people_ * num_psp_, dim_score_, 1, 1);
			if(if_search_knn_)
				top[1]->Reshape(K_ * num_people_ * num_psp_, dim_score_, 1, 1);
		} else {
			// Output feature
			top[0]->Reshape(num_people_ * num_psp_, dim_feature_, 1, 1);
			if(if_search_knn_)
				top[1]->Reshape(K_ * num_people_ * num_psp_, dim_feature_, 1, 1);
		}
		diffmap_org_to_bottom_.resize(num_people_ * num_psp_);
		if(if_search_knn_)
			diffmap_knn_to_bottom_.resize(K_ * num_people_ * num_psp_);
		if(top.size() > 2) {
			// output search index
			top[2]->Reshape(num_people_ * num_psp_, 1, 1, 1);
			if(top.size() > 3) {
				// output knn index
				top[3]->Reshape(K_ * num_people_ * num_psp_, 1, 1, 1);
			}
		}

		if(show_debug_info_) {
			printf("============Poselet KNN Layer============\n");
			printf("\nBasic Information:\n");
			printf("\t%-20s=\t%d\n", "num_scale_", num_scale_);
			printf("\t%-20s=\t%d\n", "num_ppp_", num_ppp_);
			printf("\t%-20s=\t%d\n", "num_people_", num_people_);
			printf("\t%-20s=\t%d\n", "num_psp_", num_psp_);
			printf("\t%-20s=\t%d\n", "dim_score_", dim_score_);
			printf("\t%-20s=\t%d\n", "dim_feature_", dim_feature_);
			printf("\t%-20s=\t%d\n", "dim_featureMap_", dim_featureMap_);
			printf("\t%-20s=\t%f\n", "threshold_", threshold_);
			printf("\t%-20s=\t%d\n", "K_", K_);
		}
	}

	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Search_ORG_Poselet(bottom);
		if(show_debug_info_) {
			printf("\nSearch ORG Poselet:\n");
			for(int i = 0; i < num_people_; i ++) {
				for(int j = 0; j < num_psp_; j ++) {
					printf("\tpeople(%2d)_selected(%2d):poselet(%d)\n", i+1, j+1, 
						diffmap_org_to_bottom_[i * num_psp_ + j] + 1);
				}
			}
		}
		if(if_search_knn_) {
			Search_KNN_Poselet(bottom);
			if(show_debug_info_) {
				printf("\nSearch KNN Poselet:\n");
				for(int i = 0; i < num_people_; i ++) {
					for(int j = 0; j < num_psp_; j ++) {
						printf("\tpeople(%2d)_selected(%2d):poselet(%d)\n", i+1, j+1, 
							diffmap_org_to_bottom_[i * num_psp_ + j]+1);
						for(int k = 0; k < K_; k ++) {
							// printf("\t%d", diffmap_knn_to_bottom_[(i * num_psp_ + j) * K_ + k]);
							int knn_idx = diffmap_knn_to_bottom_[(i * num_psp_ + j) * K_ + k];
							printf("\t\tk=%d, people(%2d)_poselet(%2d)_scale(%d)\n", k+1,
								(knn_idx / num_ppp_) % num_people_ + 1,
								knn_idx + 1,
								knn_idx / (num_people_ * num_ppp_) + 1);
						}
					}
				}
			}
		}
		// Copy Data
		if(show_debug_info_) printf("\nFoward...");
		const Dtype* data_bottom;
		Dtype* data_top;
		int data_dim;		
		if(this->layer_param_.poselet_knn_param().output_mode() == PoseletKNNParameter_OutputMode_OUTPUT_SCORE) {
			data_bottom = bottom[0]->cpu_data();
			data_dim = dim_score_;
		} else {
			data_bottom = bottom[2]->cpu_data();
			data_dim = dim_feature_;
		}
		data_top = top[0]->mutable_cpu_data();
		for(int i = 0; i < num_people_ * num_psp_; i ++) {
			caffe_copy(data_dim, data_bottom + data_dim * diffmap_org_to_bottom_[i], 
				data_top + data_dim * i);
		}
		if(if_search_knn_) {
			data_top = top[1]->mutable_cpu_data();
			for(int i = 0; i < num_people_ * num_psp_ * K_; i ++) {
				caffe_copy(data_dim, data_bottom + data_dim * diffmap_knn_to_bottom_[i],
					data_top + data_dim * i);
			}
		}

		if(top.size() > 2) {
			for(int i = 0; i < num_people_ * num_psp_; i ++)
				top[2]->mutable_cpu_data()[i] = static_cast<Dtype>(diffmap_org_to_bottom_[i]);
		}
		if(top.size() > 3) {
			for(int i = 0; i < K_ * num_people_ * num_psp_; i ++) {
				top[3]->mutable_cpu_data()[i] = static_cast<Dtype>(diffmap_knn_to_bottom_[i]);
			}
		}
		if(show_debug_info_) printf("done!\n");
	}

	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      	if (!propagate_down[0]) {
            return;
        }
        CHECK_EQ(top[0]->num(), diffmap_org_to_bottom_.size());
        CHECK_EQ(top[0]->num(), num_people_ * num_psp_);
        if(if_search_knn_) {
	        CHECK_EQ(top[1]->num(), diffmap_knn_to_bottom_.size());
	    	CHECK_EQ(top[1]->num(), K_ * num_people_ * num_psp_);
	    }	        
        
        if(this->layer_param_.poselet_knn_param().output_mode() == PoseletKNNParameter_OutputMode_OUTPUT_SCORE) {
        	// Output Score
        	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
        	CHECK_EQ(top[0]->channels(), dim_score_);
        	for(int i = 0; i < top[0]->num(); i ++) {
        		caffe_copy(dim_score_, top[0]->cpu_diff() + i * dim_score_, 
        			bottom[0]->mutable_cpu_diff() + diffmap_org_to_bottom_[i] * dim_score_);
        	}
        	if(if_search_knn_) {
	        	CHECK_EQ(top[1]->channels(), dim_score_);
	        	for(int i = 0; i < top[1]->num(); i ++) {
	        		caffe_axpy(dim_score_, Dtype(1), top[1]->cpu_diff() + i * dim_score_,
	        			bottom[0]->mutable_cpu_diff() + diffmap_knn_to_bottom_[i] * dim_score_);
	        	}
	        }
        } else {
        	//Output Feature
        	caffe_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_cpu_diff());
        	CHECK_EQ(top[0]->channels(), dim_feature_);
        	for(int i = 0; i < top[0]->num(); i ++) {
        		caffe_copy(dim_feature_, top[0]->cpu_diff() + i * dim_feature_,
        			bottom[2]->mutable_cpu_diff() + diffmap_org_to_bottom_[i] * dim_feature_);
        	}
        	if(if_search_knn_) {
        		CHECK_EQ(top[1]->channels(), dim_feature_);
	        	for(int i = 0; i < top[1]->num(); i ++) {
	        		caffe_axpy(dim_feature_, Dtype(1), top[1]->cpu_diff() + i * dim_feature_,
	        			bottom[2]->mutable_cpu_diff() + diffmap_knn_to_bottom_[i] * dim_feature_);
	        	}
        	}
        }
        if(show_debug_info_) {
	        printf("\nBackward:\n");
	        const Dtype* source = (this->layer_param_.poselet_knn_param().output_mode() == PoseletKNNParameter_OutputMode_OUTPUT_SCORE)?
		        bottom[0]->cpu_diff() : bottom[2]->cpu_diff();
		    int dim = (this->layer_param_.poselet_knn_param().output_mode() == PoseletKNNParameter_OutputMode_OUTPUT_SCORE)?
			    dim_score_ : dim_feature_;
			for(int i = 0; i < num_people_ * num_ppp_ * num_scale_; i++) {
				Dtype norm_diff = caffe_cpu_dot(dim, source + i * dim, source + i * dim);
				if(norm_diff > 1e-3) {
					printf("\tposelet(%d)_diff_norm:%f\n", i, sqrt(norm_diff));
				}
			}
        }
        
	}
	
	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::Search_ORG_Poselet(const vector<Blob<Dtype>*>& bottom) {
		const Dtype* score = bottom[0]->cpu_data();
		int max_idx, curr_idx;
		// CHECK_EQ(num_psp_, dim_score_) << "Only support num_psp_ == dim_score_";

		if(this->layer_param_.poselet_knn_param().poselet_weight()) {
			vector<Dtype> weight(bottom[3]->num());
			for(int i = 0; i < bottom[3]->num(); i ++)
				weight[i] = bottom[3]->cpu_data()[i * 4 + 3]; //poselet weight is the 4th channel
			for(int i = 0; i < num_people_; i ++) {
				for(int j = 0; j < num_psp_; j ++) {
					max_idx = i * num_ppp_;
					for(int p = 1; p < num_ppp_; p ++) {
						curr_idx = i * num_ppp_ + p;
						if(score[curr_idx * dim_score_ + j]  * weight[curr_idx] > 
							score[max_idx * dim_score_ + j] * weight[max_idx]) {
							max_idx = curr_idx;
						}
					}
					diffmap_org_to_bottom_[i * num_psp_ + j] = max_idx;
				}
			}
		} else {
			if(poselet_select_mode_ == 0) { // select the max score
				for(int i = 0; i < num_people_; i ++) {
					for(int j = 0; j < num_psp_; j ++) {
						max_idx = i * num_ppp_;
						for(int p = 1; p < num_ppp_; p ++) {
							curr_idx = i * num_ppp_ + p;
							if(score[curr_idx * dim_score_ + j] > score[max_idx * dim_score_ + j]) {
								max_idx = curr_idx;
							}
						}
						diffmap_org_to_bottom_[i * num_psp_ + j] = max_idx;
					}
				}	
			} else {
				// select max socre for pos_sample and min socre for neg_sample
				CHECK_GE(bottom.size(), 5);
				CHECK_EQ(bottom[4]->num(), num_people_);
				const Dtype* label_data = bottom[4]->cpu_data();
				for(int i = 0; i < num_people_; i ++) {
					for(int j = 0; j < num_psp_; j ++) {
						max_idx = i * num_ppp_;
						if(label_data[i * dim_score_ + j] > 0.5) {
							for(int p = 1; p < num_ppp_; p ++) {
								curr_idx = i * num_ppp_ + p;
								if(score[curr_idx * dim_score_ + j] > score[max_idx * dim_score_ + j]) {
									max_idx = curr_idx;
								}
							}							
						} else {
							for(int p = 1; p < num_ppp_; p ++) {
								curr_idx = i * num_ppp_ + p;
								if(score[curr_idx * dim_score_ + j] < score[max_idx * dim_score_ + j]) {
									max_idx = curr_idx;
								}
							}							
						}
						diffmap_org_to_bottom_[i * num_psp_ + j] = max_idx;
					}
				}
			}		
		}
	}

	template <typename Dtype>
	void PoseletKNNLayer<Dtype>::Search_KNN_Poselet(const vector<Blob<Dtype>*>& bottom) {
		const Dtype* feature_map = bottom[1]->cpu_data();
		Dtype squre_threshold = threshold_ * threshold_;
		int num_tar = num_people_ * num_psp_; // poselet choosed in Search_ORG_Poselet
		int num_nei = num_scale_ * num_people_ * num_ppp_;
		vector<Dtype> distance_list(K_);
		vector<Dtype> norm_list(num_nei);
		Dtype dis;
		int n, i_tar, i_prev_idx;

		for(int i = 0; i < num_nei; i++) {
			norm_list[i] = caffe_cpu_dot(dim_featureMap_, feature_map + i * dim_featureMap_,
				feature_map + i * dim_featureMap_);
		}

		for(int i_tar_idx = 0; i_tar_idx < num_tar; i_tar_idx ++) {
			i_tar = diffmap_org_to_bottom_[i_tar_idx];
			//check if i_tar has been choosed before
			for(i_prev_idx = i_tar_idx - i_tar_idx%num_psp_; i_prev_idx < i_tar_idx; i_prev_idx++) {
				if(diffmap_org_to_bottom_[i_prev_idx] == i_tar)
					break;
			}
			if(i_prev_idx != i_tar_idx) {
				// The knn of i_tar has been searched before. So just copy the result.
				for(int k = 0; k < K_; k++) {
					diffmap_knn_to_bottom_[i_tar_idx * K_ + k] = diffmap_knn_to_bottom_[i_prev_idx * K_ + k];
				}
			}else{
				// The knn of i_tar has not been searched. Do it here.
	            for(int i_knn = 0; i_knn < K_; i_knn ++) {
					diffmap_knn_to_bottom_[i_tar_idx * K_+ i_knn] = i_tar; // default neighbor is the target itself
					distance_list[i_knn] = squre_threshold;
				}
				for(int i_nei = 0; i_nei < num_nei; i_nei ++) {
					if((this->layer_param_.poselet_knn_param().knn_filter_level() >= 1) && 
						(i_nei / num_ppp_ == i_tar / num_ppp_))
						continue;// filter same person in standard scale
					if((this->layer_param_.poselet_knn_param().knn_filter_level() >= 2) &&
						(i_nei % (num_people_ * num_ppp_) == i_tar))
						continue;// filter same poselet in all scales
					if((this->layer_param_.poselet_knn_param().knn_filter_level() >= 3) && 
						((i_nei / num_ppp_) % num_people_ == i_tar / num_ppp_))
						continue;// filter same person in all scales
					
					dis = norm_list[i_tar] + norm_list[i_nei] - 2 * caffe_cpu_dot(dim_featureMap_,
						feature_map + i_tar * dim_featureMap_, feature_map + i_nei * dim_featureMap_);
					// if(show_debug_info_) printf("poselet_dis(%d, %d)=%f\n", i_tar, i_nei, sqrt(dis));
					
					n = 0;
					while(n < K_) {
						if(dis > distance_list[n]) {
							break;
						}
						n ++;
					}
					if(n > 0) {
						for(int i = 0; i < n - 1; i ++) {
							distance_list[i] = distance_list[i + 1];
							diffmap_knn_to_bottom_[i_tar_idx * K_ + i] = diffmap_knn_to_bottom_[i_tar_idx * K_ + i + 1];
						}
						distance_list[n - 1] = dis;
						diffmap_knn_to_bottom_[i_tar_idx * K_ + n - 1] = i_nei;
					}
				}
			}
		}
	}
    

#ifdef CPU_ONLY
	STUB_GPU(PoseletKNNLayer);
#endif

INSTANTIATE_CLASS(PoseletKNNLayer);
REGISTER_LAYER_CLASS(PoseletKNN);

}  // namespace caffe
