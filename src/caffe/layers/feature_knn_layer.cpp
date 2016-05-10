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

// #if _MSC_VER < 1800
// inline double round(double x) {
// 	return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
// }
// #endif

namespace caffe {

	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		this->show_debug_info_ = this->layer_param_.feature_knn_param().show_debug_info();
		// if(show_debug_info_) printf("FeatureKNNLayer: LayerSetUp...");

		threshold_ = this->layer_param_.feature_knn_param().knn_threshold();
		K_ = this->layer_param_.feature_knn_param().k_neighbor_num();
		CHECK_GT(K_, 0);
		// if(show_debug_info_) printf("done!\n");
	}

	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top) {
		// if(show_debug_info_) printf("FeatureKNNLayer: Reshape\n");
		vector<int> shape(2);
		shape[0] = bottom[0]->num() / 4;// 4 is the number of image pyramid 
		shape[1] = bottom[0]->channels();
		top[0]->Reshape(shape);
		shape[0] *= K_;
		top[1]->Reshape(shape);
		// if(show_debug_info_) {
		// 	printf("num: %d, channels: %d, K: %d\n", shape[0]/K_, shape[1], K_);
		// 	printf("FeatureKNNLayer: Reshape done!\n");
		// }
	}
	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::ReshapeOnTheFly(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// if(show_debug_info_) printf("FeatureKNNLayer: Reshape_otf\n");
		GetImageInfo(bottom);
		// if(show_debug_info_) printf("GetImageInfo done!\n");
		vector<int> shape(2);
		shape[0] = num_people_;
		shape[1] = dim_;
		top[0]->Reshape(shape);
		shape[0] = num_people_ * K_;
		top[1]->Reshape(shape);
		diffmap_top0_to_bottom_.resize(num_people_);
		diffmap_top1_to_bottom_.resize(num_people_ * K_);
		
		for(int i = 0; i < num_people_; i ++) {
			diffmap_top0_to_bottom_[i] = 0;
		}
		for (int i = 0; i < num_people_ * K_; i ++) {
			diffmap_top1_to_bottom_[i] = 0;
		}
		if(show_debug_info_) {
			printf("############# %s ##############\n", this->layer_param_.feature_knn_param().debug_info().c_str());
			printf("num_people: %d\n", num_people_);			
		}
	}

	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::SearchKNN_cpu(const Dtype* image_data, int num_people, int num_scale, int dim,
		int k, Dtype threshold, vector<vector<int> >& knn_list, Blob<Dtype>& buffer) {
		// if(show_debug_info_) printf("SearchKNN...\n");
		
		int num_people_total = num_people * num_scale;
		Dtype squre_threshold = threshold * threshold;
		knn_list.resize(num_people);
		vector<float> distance_list(k);
		vector<float> norm_list(num_people_total);
		// Compute the norm of each feature
		
		for(int i = 0; i < num_people_total; i++) {
			norm_list[i] = caffe_cpu_dot(dim, image_data + i * dim, image_data + i * dim);
		}

		// Compute the distance between each two features
		for(int i_tar = 0; i_tar < num_people; i_tar ++) {
			knn_list[i_tar].resize(k);
			for(int i_knn = 0; i_knn < k; i_knn++) {
				knn_list[i_tar][i_knn] = i_tar; // default neiboor is the target itself
				distance_list[i_knn] = squre_threshold;
			}

			for(int i_nei = 0; i_nei < num_people_total; i_nei ++) {
				if(i_nei == i_tar) {
					continue;
				}
				// Compute distance
				Dtype dis = norm_list[i_tar] + norm_list[i_nei] -
				    2 * caffe_cpu_dot(dim, image_data + i_tar * dim, image_data + i_nei * dim);
	
				if(show_debug_info_) {
					Dtype cos_dis = 1- sqrt(caffe_cpu_dot(dim, image_data + i_tar * dim, image_data + i_nei * dim) 
							/ ((norm_list[i_tar] * norm_list[i_nei])));
					printf("distance(%d,%d): %f\t", i_tar, i_nei, sqrt(dis));
					printf("cos_dist(%d,%d): %f\n", i_tar, i_nei, cos_dis);

				}
				
				// Sort
				int n = 0;
				while(n < k) {
					if(dis > distance_list[n])
						break;
					n ++;
				}

				if(n > 0) {
					for(int i = 0; i < n - 1; i ++) {
						distance_list[i] = distance_list[i + 1];
						knn_list[i_tar][i] = knn_list[i_tar][i + 1];
					}
					distance_list[n - 1] = dis;
					knn_list[i_tar][n - 1] = i_nei;
				}
		    }
		}
		if(show_debug_info_) {
			printf("KNN Search Result:\n");
			printf("num_people:%d, num_knn_cand:%d\n", num_people, num_people_total);
			for(int i = 0; i < num_people; i ++) {
				for(int j = 0; j < k; j ++) {
					printf("target_%d ,knn_%d: %d\n", i, j, knn_list[i][j]);
				}									
			}
		}
	}

	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

        int first_id_in_image = 0;
        vector<vector<int> > knn_list;
		this->ReshapeOnTheFly(bottom, top);
		//if(show_debug_info_) printf("FeatureKNN: Forward...");
		// if(show_debug_info_) {
		// 	int num = bottom[0]->num();
		// 	int dim = bottom[0]->channels();
		// 	printf("\n feature_knn_layer: Forward_cpu\n");
		// 	printf("feature_number: %d    feature_length:%d\n", num, dim);
		// 	CHECK_EQ(bottom[0]->num(), bottom[1]->num());

		// 	// printf("feature_data: \n");
		// 	// for(int i = 0; i < num; i ++) {
		// 	// 	for(int j = 0; j < dim; j++)
		// 	// 		printf("%.3f\t", bottom[0]->cpu_data()[i * dim + j]);
		// 	// 	printf("\n");
		// 	// }
		// 	printf("poselet_info: \n");
		// 	for(int i = 0; i < num; i ++) {
		// 		for(int j = 0; j < bottom[1]->channels(); j ++)
		// 			printf("%.3f\t", bottom[1]->cpu_data()[i * bottom[1]->channels() + j]);
		// 		printf("\n");
		// 	}
		// 	printf("image number: %d\n", num_image_);
		// }
		
		for(int i_image = 0; i_image < num_image_; i_image ++) {
			SearchKNN_cpu(bottom[0]->cpu_data() + first_id_in_image * dim_ * num_scale_,
				          people_num_in_image_[i_image],
				          num_scale_,
				          dim_,
				          K_,
				          threshold_,
				          knn_list,
				          buffer_);
		    caffe_copy(people_num_in_image_[i_image] * dim_, 
		    	bottom[0]->cpu_data() + first_id_in_image * dim_ * num_scale_,
		    	top[0]->mutable_cpu_data() + first_id_in_image * dim_);
		    for(int i_tar = 0; i_tar < people_num_in_image_[i_image]; i_tar ++) {
		    	diffmap_top0_to_bottom_[first_id_in_image + i_tar] = 
		    	    first_id_in_image * num_scale_ + i_tar;

		    	for(int i_knn = 0; i_knn < K_; i_knn++) {
		    		caffe_copy(dim_, 
		    			bottom[0]->cpu_data() + (first_id_in_image * num_scale_ + knn_list[i_tar][i_knn]) * dim_,
		    			top[1]->mutable_cpu_data() + ((first_id_in_image + i_tar) * K_ + i_knn) * dim_);

		    	    diffmap_top1_to_bottom_[(first_id_in_image + i_tar) * K_ + i_knn] =
                        first_id_in_image * num_scale_ + knn_list[i_tar][i_knn];
		    	}
		    }

		    first_id_in_image += people_num_in_image_[i_image];
		}
		//if(show_debug_info_) printf("Done!\n");
	}


	template <typename Dtype>
	void FeatureKNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      	//if(show_debug_info_) printf("FeatureKNN: Backward...");
      	if (!propagate_down[0]) {
            return;
        }       
      	CHECK_EQ(top[0]->num(), diffmap_top0_to_bottom_.size());
      	CHECK_EQ(top[1]->num(), diffmap_top1_to_bottom_.size());
      	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

      	for(int i = 0; i < top[0]->num(); i ++) {
      		caffe_copy(dim_, top[0]->cpu_diff() + i * dim_, 
      			bottom[0]->mutable_cpu_diff() + diffmap_top0_to_bottom_[i] * dim_);
      	}

      	for(int i = 0; i < top[1]->num(); i++) {
      		caffe_axpy(dim_, Dtype(1), top[1]->cpu_diff() + i * dim_,
      			bottom[0]->mutable_cpu_diff() + diffmap_top1_to_bottom_[i] * dim_);
      	}
      	//if(show_debug_info_) printf("done\n");
	}
    
    // Set num_scale_, num_image_, dim_, num_people_, people_num_in_image_
    template <typename Dtype>
    void FeatureKNNLayer<Dtype>::GetImageInfo(const vector<Blob<Dtype>*>& bottom) {
    	//if(show_debug_info_) printf("FeatureKNN: GetImageInfo\n");
    	const Dtype* poselet_info = bottom[1]->cpu_data();
    	int num_poselet = bottom[1]->num();
    	int dim_poselet_info = bottom[1]->channels(); // should be 2 ([image_id, person_id])
    	int image_id = 0;
    	people_num_in_image_.clear();
    	for(int i = 0; i < num_poselet; i++) {
    		image_id = poselet_info[i * dim_poselet_info];
    		if(image_id == people_num_in_image_.size()) {
    			people_num_in_image_.push_back(1);
    		} else {
    			people_num_in_image_[image_id] ++;
    		}
    	}
    	
    	CHECK(people_num_in_image_[image_id] % 
    		static_cast<int>(poselet_info[dim_poselet_info * num_poselet - 1] + 1) == 0);
    	num_scale_ = people_num_in_image_[image_id] / 
    	    static_cast<int>(poselet_info[dim_poselet_info * num_poselet - 1] + 1);
    	num_image_ = image_id + 1;
    	dim_ = bottom[0]->channels();
    	num_people_ = num_poselet / num_scale_;

    	for(int i = 0; i < num_image_; i ++) {
    		people_num_in_image_[i] /= num_scale_;
    	}
    	//if(show_debug_info_) printf("num_poselet: %d\ndim_poselet_info: %d\nnum_image: %d\nnum_people: %d\nnum_scale: %d\n", 
    	//	num_poselet, dim_poselet_info, num_image_, num_people_, num_scale_);
    }

#ifdef CPU_ONLY
	STUB_GPU(FeatureKNNLayer);
#endif

INSTANTIATE_CLASS(FeatureKNNLayer);
REGISTER_LAYER_CLASS(FeatureKNN);

}  // namespace caffe
