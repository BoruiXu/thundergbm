//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/hist_cut.h"
#include "thundergbm/quantile_sketch.h"
#include "thundergbm/syncarray.h"
#include <sstream>
#include <omp.h>
#include <thundergbm/hist_cut.h>
#include <thundergbm/util/cub_wrapper.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <climits>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/unique.h"

#include <chrono>
#include<iostream>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()
/**
 * not fast but need less memory
 */
void HistCut::get_cut_points2(SparseColumns &columns, int max_num_bins, int n_instances) {
    LOG(INFO) << "Getting cut points...";
    int n_column = columns.n_column;
    SyncArray<float> unique_vals(n_column * n_instances);
    SyncArray<int> tmp_row_ptr(n_column + 1);

    SyncArray<int> tmp_params(2);  // [0] --> num_cut_points, [1] --> max_num_bins
    int h_tmp_params[2] = {0, max_num_bins};
    tmp_params.copy_from(h_tmp_params, 2);

    // get the original csc data
    auto csc_val_data = columns.csc_val.device_data();
    auto csc_col_ptr_data = columns.csc_col_ptr.device_data();
    auto unique_vals_data = unique_vals.device_data();
    auto tmp_row_ptr_data = tmp_row_ptr.device_data();
    auto tmp_params_data = tmp_params.device_data();

    // start to get cut points
    device_loop(n_column, [=] __device__(int fid){
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid+1] - col_start;
        auto val_data = csc_val_data + col_start;
        auto unique_start = unique_vals_data + fid*n_instances;  // notice here
        int unique_len = thrust::unique_copy(thrust::device, val_data, val_data + col_len, unique_start) - unique_start;
        int n_cp = (unique_len <= tmp_params_data[1]) ? unique_len : tmp_params_data[1];
        tmp_row_ptr_data[fid+1] = unique_len;
        atomicAdd(&tmp_params_data[0], n_cp);
    });

    // merge the cut points
    tmp_params_data = tmp_params.host_data();
    cut_points_val.resize(tmp_params_data[0]);
    cut_row_ptr.resize(n_column + 1);
    cut_fid.resize(tmp_params_data[0]);

    cut_row_ptr.copy_from(tmp_row_ptr);
    auto cut_row_ptr_data = cut_row_ptr.host_data();
    tmp_row_ptr_data = tmp_row_ptr.host_data();
    for(int i = 1; i < (n_column + 1); i++) {
        if(tmp_row_ptr_data[i] <= tmp_params_data[1])
            cut_row_ptr_data[i] += cut_row_ptr_data[i-1];
        else
            cut_row_ptr_data[i] = cut_row_ptr_data[i-1] + max_num_bins;
    }

    auto cut_point_val_data = cut_points_val.device_data();
    tmp_row_ptr_data = tmp_row_ptr.device_data();
    tmp_params_data = tmp_params.device_data();
    cut_row_ptr_data = cut_row_ptr.device_data();
    unique_vals_data = unique_vals.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__ (int fid, int i){
        // fid --> [0, n_column)  &&  i -->> [cut_row_ptr[fid], cut_row_ptr[fid+1])
        int unique_len = tmp_row_ptr_data[fid+1];
        int unique_idx =  i - cut_row_ptr_data[fid];
        int cp_idx = (unique_len <= tmp_params_data[1]) ? unique_idx : (unique_len / tmp_params_data[1] * unique_idx);
        cut_point_val_data[i] = unique_vals_data[fid*n_instances + cp_idx];
    });

    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
}

template<typename T>
void syncarray_resize(SyncArray<T> &buf_array, int new_size) {
    CHECK_GE(buf_array.size(), new_size) << "The size of the target Syncarray must greater than the new size. ";
    SyncArray<T> tmp_array(new_size);
    tmp_array.copy_from(buf_array.device_data(), new_size);
    buf_array.resize(new_size);
    buf_array.copy_from(tmp_array);
    tmp_array.resize(0);
}

void unique_by_flag(SyncArray<float> &target_arr, SyncArray<int> &flags, int n_columns) {
    using namespace thrust::placeholders;
    //double type array
    //SyncArray<double> target_arr_double;
    //target_arr_double.resize(target_arr.size());
    //auto target_arr_double_device = target_arr_double.device_data();
    //auto target_arr_device = target_arr.device_data();

    //device_loop(target_arr.size(),[=]__device__(size_t i){
    //    target_arr_double_device[i] = (double)target_arr_device[i];
    //});

    double max_elem = max_elements(target_arr);
    LOG(DEBUG) << "max feature value: " << max_elem;
    CHECK_LT(max_elem + n_columns*(max_elem + 1),INT_MAX) << "Max_values is too large to be transformed";
        
    //double type array
    SyncArray<double> target_arr_double;
    target_arr_double.resize(target_arr.size());
    auto target_arr_double_device = target_arr_double.device_data();
    auto target_arr_device = target_arr.device_data();

    device_loop(target_arr.size(),[=]__device__(size_t i){
        target_arr_double_device[i] = (double)target_arr_device[i];
    });
    //new double method
    // 1. transform data into unique ranges
    thrust::transform(thrust::device,
                      target_arr_double.device_data(),
                      target_arr_double.device_end(),
                      flags.device_data(),
                      target_arr_double.device_data(),
                      (_1 + _2 * (max_elem + 1)));
    // 2. sort the transformed data
    sort_array(target_arr_double, false);
    thrust::reverse(thrust::device, flags.device_data(), flags.device_end());
    // 3. eliminate duplicates
    auto new_end = thrust::unique_by_key(thrust::device, target_arr_double.device_data(), target_arr_double.device_end(),
                                         flags.device_data());
    int new_size = new_end.first - target_arr_double.device_data();
    syncarray_resize(target_arr, new_size);
    //syncarray_resize(target_arr_double, new_size);
    syncarray_resize(flags, new_size);
    // 4. transform data back
    thrust::transform(thrust::device, target_arr_double.device_data(),
                      target_arr_double.device_data()+new_size,
                      flags.device_data(),
                      target_arr_double.device_data(),
                      (_1 - _2 * (max_elem + 1)));
    cub_sort_by_key(flags, target_arr_double);

    device_loop(target_arr.size(),[=]__device__(size_t i){

        target_arr_device[i] = (float)target_arr_double_device[i];

    });
    target_arr_double.resize(0);
    //SyncMem::clear_cache(); 
    
}

//define sort opeartion
typedef thrust::tuple<float, int> float_int;
struct Op {
  __device__ bool operator()(const float_int &a, const float_int &b) {
    if (thrust::get<1>(a) == thrust::get<1>(b)) {
        return thrust::get<0>(a) < thrust::get<0>(b);
    }
    return thrust::get<1>(a) < thrust::get<1>(b);
  }
};
//new func
void unique_by_flag2 (SyncArray<float_type> &target_arr, SyncArray<int> &flags, int n_columns){

    auto traget_arr_data = target_arr.device_data();
    auto flags_data = flags.device_data();
    size_t len = flags.size();
    

    //make zip
    auto zip_array = thrust::make_zip_iterator(thrust::make_tuple(traget_arr_data, flags_data));
    
    //sort
    thrust::sort(thrust::device,zip_array,zip_array+len,Op());

    //unique
    auto new_end = thrust::unique(thrust::device,zip_array,zip_array+len);

    //new size 
    size_t new_size = new_end-zip_array;

    syncarray_resize(target_arr, new_size);
    syncarray_resize(flags, new_size);


}
/**
 * fast but cost more memory
 */
void HistCut::get_cut_points3(SparseColumns &columns, int max_num_bins, int n_instances) {
    LOG(INFO) << "Fast getting cut points...";
    int n_column = columns.n_column;
    //cut_points_val.resize(columns.csc_val_origin.size());
    //cut_row_ptr.resize(columns.csc_col_ptr_origin.size());
    //cut_fid.resize(columns.csc_val_origin.size());
    //cut_points_val.copy_from(columns.csc_val_origin);
   
    //size_t block_num = (columns.csc_val_origin.size()/n_column - 1)/256+1;

    //auto cut_fid_data = cut_fid.device_data();
    //device_loop_2d(n_column, columns.csc_col_ptr_origin.device_data(), [=] __device__(int fid, int i) {
    //    cut_fid_data[i] = fid;
    //},block_num);
    
    cut_points_val.resize(columns.csr_val.size());
    cut_row_ptr.resize(n_column+1);
    cut_fid.resize(columns.csr_val.size());
    cut_points_val.copy_from(columns.csr_val);
    cut_fid.copy_from(columns.csr_col_idx);

    auto cut_fid_data = cut_fid.device_data();
    size_t block_num = 1;
    

    unique_by_flag2(cut_points_val, cut_fid, n_column);

    cut_row_ptr.resize(n_column + 1);
    auto cut_row_ptr_data = cut_row_ptr.device_data();
    device_loop(cut_fid.size(), [=] __device__(int fid) {
        atomicAdd(cut_row_ptr_data + cut_fid_data[fid] + 1, 1);
    });
    thrust::inclusive_scan(thrust::device, cut_row_ptr_data, cut_row_ptr_data + cut_row_ptr.size(), cut_row_ptr_data);

    size_t block_num2 = 1+ (cut_points_val.size()/n_column)/256;
    SyncArray<int> select_index(cut_fid.size());
    auto select_index_data = select_index.device_data();
    device_loop_2d_with_maximum(n_column, cut_row_ptr_data, max_num_bins, [=] __device__(int fid, int i, int interval) {
        int feature_idx = i - cut_row_ptr_data[fid];
        if(interval == 0)
            select_index_data[i] = 1;
        else if(feature_idx < max_num_bins)
            select_index_data[cut_row_ptr_data[fid] + interval * feature_idx] = 1;
    },block_num2);

    cub_select(cut_fid, select_index);
    cub_select(cut_points_val, select_index);
    
    
    cut_fid_data = cut_fid.device_data();
    cut_row_ptr.resize(n_column + 1);
    cut_row_ptr_data = cut_row_ptr.device_data();
    device_loop(cut_fid.size(), [=] __device__(int fid) {
        atomicAdd(cut_row_ptr_data + cut_fid_data[fid] + 1, 1);
    });
    thrust::inclusive_scan(thrust::device, cut_row_ptr_data, cut_row_ptr_data + cut_row_ptr.size(), cut_row_ptr_data);
    SyncMem::clear_cache();

    //test xgboost cut
    auto ptr_host = cut_row_ptr.host_data();
    std::vector<int> tmp1;
    std::ifstream infile1("/home/xbr/hpc_code/xgboost/test_bench/ptr.txt");
    int val1;
    while(infile1 >> val1){
        tmp1.push_back(val1);
    }
    infile1.close();
    //cut_row_ptr.resize(tmp1.size());
    //cut_row_ptr.copy_from(tmp1.data(),tmp1.size());
    for(int i =0;i<cut_row_ptr.size();i++){
        ptr_host[i] = tmp1[i+1]-1;
    }

    std::vector<float> tmp2;
    std::ifstream infile2("/home/xbr/hpc_code/xgboost/test_bench/val.txt");
    float val2;
    while(infile2 >> val2){
        tmp2.push_back(val2);
    }
    infile2.close();
    cut_points_val.resize(tmp2.size()-1);
    //cut_points_val.copy_from(tmp2.data(),tmp2.size());
    auto val_host = cut_points_val.host_data();
    
    for(int i = 0;i<cut_row_ptr.size()-1;i++){
        int s = ptr_host[i];
        int e = ptr_host[i+1];

        int s2 = tmp1[i+1];
        int e2 = tmp1[i+2];

        for(int j = s;j<e;j--){
            val_host[j] = tmp2[s2];
            s2++;
        }
    }
    cut_fid.resize(cut_points_val.size());
    auto fid_host = cut_fid.host_data();
    //auto ptr_host = cut_row_ptr.host_data();

    for(int i = 0;i<cut_row_ptr.size()-1;i++){
        int s = ptr_host[i];
        int e = ptr_host[i+1];
        for(int j = s;j<e;j++){
            fid_host[j] = i;
        }
    } 

    LOG(DEBUG) << "--->>>>  cut points value: " << cut_points_val;
    LOG(DEBUG) << "--->>>> cut row ptr: " << cut_row_ptr;
    LOG(DEBUG) << "--->>>> cut fid: " << cut_fid;
    LOG(INFO) << "TOTAL CP:" << cut_fid.size();
    LOG(INFO) << "NNZ: " << columns.csr_val.size();
}
