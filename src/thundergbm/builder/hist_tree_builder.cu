//
// Created by ss on 19-1-20.
//
#include "thundergbm/builder/hist_tree_builder.h"

#include "thundergbm/util/cub_wrapper.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/sequence.h"
#include "thrust/binary_search.h"
#include "thundergbm/util/multi_device.h"

#include "thundergbm/util/sp_trans.h"
#include "omp.h"
#include "cusparse.h"
#include <chrono>
#include<iostream>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()

extern long long total_sort_time_hist;
extern long long total_time_hist1;


void check_hist_res(GHPair* hist, GHPair* hist_test, int n_bins){

    //check result
    float avg_diff_g = 0;
    float total_diff_g= 0;
    
    for(int i = 0;i<n_bins;++i){
        
        total_diff_g = abs(hist_test[i].g-hist[i].g);
    
    }
    avg_diff_g = total_diff_g/n_bins;
    
    LOG(INFO)<<"total diff g is "<<total_diff_g<<" avg diff g is "<<avg_diff_g;

}

void csc2csr(SyncArray<int> &csc_col_ptr,SyncArray<int> &csc_row_idx,SyncArray<float_type> &csc_val,
            SyncArray<int> &row_ptr,SyncArray<int> &col_idx,SyncArray<float_type> &val, 
            int n_instance, int n_feature){
    
    LOG(INFO)<<"run csc to csr...";
    //SyncArray<unsigned char> val;
    //SyncArray<int> val1;
    //SyncArray<int> val2;
    int nnz = csc_row_idx.size();

    //val1.resize(nnz);
    //val2.resize(nnz);
    val.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(n_instance+1);

    //using cusparse convert csc to csr

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    //char
    cudaDataType data_type = CUDA_R_32F;//other datatype cause wrong?
    cusparseStatus_t status;

    size_t buffer_size = 1;
    status = cusparseCsr2cscEx2_bufferSize(handle, n_feature, n_instance, nnz, 
                                    csc_val.device_data(),csc_col_ptr.device_data(),csc_row_idx.device_data(),
                                    val.device_data(),row_ptr.device_data(),col_idx.device_data(),
                                    data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_size);
    SyncArray<char> tmp_buffer(buffer_size);
    LOG(INFO)<<"buffer size is "<<buffer_size/1e9<<" G";
    cusparseCsr2cscEx2(handle, n_feature, n_instance, nnz, 
                            csc_val.device_data(),csc_col_ptr.device_data(),csc_row_idx.device_data(),
                            val.device_data(),row_ptr.device_data(),col_idx.device_data(),
                            data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, tmp_buffer.device_data());

    cudaDeviceSynchronize();
    LOG(INFO)<<"return status is "<<status;
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    tmp_buffer.resize(0);
    SyncMem::clear_cache();
    //csc_val.resize(0);
    //csc_col_ptr.resize(0);
    //csc_row_idx.resize(0);


}

//csc to csr on cpu
void csc2csr_cpu(int *host_csc_col_ptr,int *host_csc_row_idx,float_type *host_csc_val,
            int *host_row_ptr,int *host_col_idx,float_type *host_val, 
            int n_instance, int n_feature,size_t nnz){
    LOG(INFO)<<"run csc to csr in cpu...";
    
    //initialize row_ptr
    //memset 0
    memset(host_row_ptr,0,sizeof(int)*(n_instance+1));
    int *tmp_loc = new int[nnz];
    #pragma omp parallel for
    for(int i=0;i<nnz;i++){
        int idx = host_csc_row_idx[i]+1;
        //tmp_loc[i] = host_row_ptr[idx];
        
        #pragma omp atomic capture
        tmp_loc[i] = host_row_ptr[idx]++;
    }

    for(int i=1;i<n_instance+1;i++){
        host_row_ptr[i] += host_row_ptr[i-1];
    }

    //parallel construct csr_col and csr_val

    #pragma omp parallel for
    for(int i=0;i<n_feature;i++){
        int start = host_csc_col_ptr[i];
        int end = host_csc_col_ptr[i+1];
        for(int j=start;j<end;j++){
            int idx = host_csc_row_idx[j];
            int pos = host_row_ptr[idx]+tmp_loc[j];
            host_col_idx[pos] = i;
            host_val[pos] = host_csc_val[j];
        }
    }
}

void HistTreeBuilder::get_bin_ids() {
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        SparseColumns &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        //auto &dense_bin_id = this->dense_bin_id[device_id];
        using namespace thrust;
        int n_column = columns.n_column;
        size_t nnz = columns.nnz;
        auto cut_row_ptr = cut.cut_row_ptr.device_data();
        auto cut_points_ptr = cut.cut_points_val.device_data();
        
        int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);
        //original order csc
        //auto csc_val_origin_data = columns.csc_val_origin.device_data();
        
        //auto &bin_id_origin = this->bin_id_origin[device_id];
        //bin_id_origin.resize(columns.nnz);
        //auto bin_id_origin_data = bin_id_origin.device_data();
        
        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_col_idx = this->csr_col_idx[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];

        //set poniter
        csr_row_ptr.resize(n_instances+1);
        csr_col_idx.resize(nnz);
        csr_bin_id.resize(nnz);
        
        csr_row_ptr.set_device_data(columns.csr_row_ptr.device_data());
        csr_col_idx.set_device_data(columns.csr_col_idx.device_data());

        {
            auto lowerBound = [=]__device__(const float_type *search_begin, const float_type *search_end, float_type val) {
                const float_type *left = search_begin;
                const float_type *right = search_end - 1;

                while (left != right) {
                    const float_type *mid = left + (right - left) / 2;
                    if (*mid <= val)
                        right = mid;
                    else left = mid + 1;
                }
                return left;
            };
            TIMED_SCOPE(timerObj, "binning");
            //for original order csc
            //device_loop_2d(n_column, columns.csc_col_ptr_origin.device_data(), [=]__device__(int cid, int i) {
            //    auto search_begin = cut_points_ptr + cut_row_ptr[cid];
            //    auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
            //    auto val = csc_val_origin_data[i];
            //    bin_id_origin_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            //}, n_block);

            //get csr bin id
            auto csr_col_idx_data = csr_col_idx.device_data();
            auto csr_val_data = columns.csr_val.device_data();
            auto csr_bin_id_data = csr_bin_id.device_data();
            device_loop_2d(n_instances, csr_row_ptr.device_data(), [=]__device__(int instance_id, int i) {
                auto cid = csr_col_idx_data[i];
                auto search_begin = cut_points_ptr + cut_row_ptr[cid];
                auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
                auto val = csr_val_data[i];
                csr_bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin + cut_row_ptr[cid];
            }, n_block);

        }
        //csc2csr
        //get rest gpu mem 
        //TODO when on multi devices this should be modified
        
        //size_t free, total,dataset_size;
        //cudaMemGetInfo(&free, &total);
        //free = 1.0*free/1e9;
        //dataset_size = (nnz*4*2+n_column*4)/1e9;
        //LOG(INFO)<<"free mem is "<<free<<" G";
        //LOG(INFO)<<"csc dataset size is "<<dataset_size<<" G";
        //if(2.6*dataset_size>free){
        //    use_gpu = false;
        //}
        //if(use_gpu){
        //    csc2csr(columns.csc_col_ptr_origin,columns.csc_row_idx_origin,bin_id_origin,
        //            csr_row_ptr,csr_col_idx,csr_bin_id,
        //            n_instances,n_column);
        //}
        //else{
        //    //set size 
        //    csr_row_ptr.resize(n_instances+1);
        //    csr_col_idx.resize(nnz);
        //    csr_bin_id.resize(nnz);
        //    csc2csr_cpu(columns.csc_col_ptr_origin.host_data(),columns.csc_row_idx_origin.host_data(),bin_id_origin.host_data(),
        //                csr_row_ptr.host_data(),csr_col_idx.host_data(),csr_bin_id.host_data(),
        //                n_instances,n_column,nnz);
        //    //LOG(INFO)<<"csc to csr using cpu..."; 
        //    //sptrans_scanTrans<int,float_type>(n_column, n_instances, nnz,
        //    //                                columns.csc_col_ptr_origin.host_data(), columns.csc_row_idx_origin.host_data(), bin_id_origin.host_data(), 
        //    //                                csr_col_idx.host_data(), csr_row_ptr.host_data(), csr_bin_id.host_data());
        //}
        //columns.csc_val_origin.clear_device();
        //columns.csc_val_origin.resize(0);
        columns.csr_val.resize(0);
        //csr col idx do not need
        columns.csr_col_idx.resize(0);
        SyncMem::clear_cache();
        //auto max_num_bin = param.max_num_bin;
        //
        //
        ////here 32 is the max nodes in one level
        //size_t current_dense_size = (long long)n_instances*32;

        //dense_bin_id.resize(current_dense_size);
        //auto dense_bin_id_data = dense_bin_id.device_data();
        //
        //device_loop(current_dense_size, [=]__device__(size_t i) {
        //    dense_bin_id_data[i] = max_num_bin;
        //});

    });
}

void HistTreeBuilder::find_split(int level, int device_id) {
    std::chrono::high_resolution_clock timer;

    const SparseColumns &columns = shards[device_id].columns;
    SyncArray<int> &nid = ins2node_id[device_id];
    SyncArray<GHPair> &gh_pair = gradients[device_id];
    Tree &tree = trees[device_id];
    SyncArray<SplitPoint> &sp = this->sp[device_id];
    SyncArray<bool> &ignored_set = shards[device_id].ignored_set;
    HistCut &cut = this->cut[device_id];
    //auto &dense_bin_id = this->dense_bin_id[device_id];
    auto &last_hist = this->last_hist[device_id];

    TIMED_FUNC(timerObj);
    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    size_t n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 1 << (param.depth-1);
    size_t n_max_splits = n_max_nodes * (long long)n_bins;
    size_t n_split = n_nodes_in_level * (long long)n_bins;

    int n_block = fminf((columns.nnz / this->n_instances - 1) / 256 + 1, 4 * 84); 
    //csr bin id
    auto &csr_row_ptr = this->csr_row_ptr[device_id];
    auto &csr_col_idx = this->csr_col_idx[device_id];
    auto &csr_bin_id  = this->csr_bin_id[device_id];
    
    auto csr_row_ptr_data = csr_row_ptr.device_data();
    auto csr_col_idx_data = csr_col_idx.device_data();
    auto csr_bin_id_data = csr_bin_id.device_data();
    LOG(TRACE) << "start finding split";
    TDEF(sort_time)

    //new variables
    size_t len_hist = 2*n_bins;
    size_t len_missing = 2*n_column;

    //remember resize variable to clear
    //find the best split locally
    {
        using namespace thrust;
        auto t_build_start = timer.now();

        //calculate split information for each split
        SyncArray<GHPair> hist(len_hist);
        SyncArray<GHPair> missing_gh(len_missing);
        auto cut_fid_data = cut.cut_fid.device_data();
        auto i2fid = [=] __device__(int i) { return cut_fid_data[i % n_bins]; };
        auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);
        {
            TSTART(sort_time)
            {
                TIMED_SCOPE(timerObj, "build hist");
                {
                    size_t
                    smem_size = n_bins * sizeof(GHPair);
                    LOG(DEBUG) << "shared memory size = " << smem_size / 1024.0 << " KB";
                    if (n_nodes_in_level == 1) {
                        //root
                        auto hist_data = hist.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        //auto dense_bin_id_data = dense_bin_id.device_data();
                        auto max_num_bin = param.max_num_bin;
                        auto n_instances = this->n_instances;
                        device_loop_hist_csr_root(n_instances,csr_row_ptr_data, [=]__device__(int i,int j){
                        
                            //int fid = csr_col_idx_data[j];
                            int bid = (int)csr_bin_id_data[j];
                            
                            //int feature_offset = cut_row_ptr_data[fid];
                            const GHPair src = gh_data[i];
                            //GHPair &dest = hist_data[feature_offset + bid]; 
                            GHPair &dest = hist_data[bid]; 
                            if(src.h != 0)
                                atomicAdd(&dest.h, src.h);
                            if(src.g != 0)
                                atomicAdd(&dest.g, src.g);                            
                        
                        },n_block);

                        //new code 
                        last_hist.copy_from(hist.device_data(),n_bins);
                        cudaDeviceSynchronize();

                        inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + n_bins,
                                      hist.device_data(), hist.device_data());

                        { //missing 
                            auto nodes_data = tree.nodes.device_data();
                            auto missing_gh_data = missing_gh.device_data();
                            auto cut_row_ptr = cut.cut_row_ptr.device_data();
                            auto hist_data = hist.device_data();
                            device_loop(n_column, [=]__device__(int pid) {
                                int nid0 = pid / n_column;
                                int nid = nid0 + nid_offset;
                                if (!nodes_data[nid].splittable()) return;
                                int fid = pid % n_column;
                                if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
                                    GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
                                    missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
                                }
                            });

                        }

                        //
                        SyncArray<float_type> gain(n_bins);
                        {
                //            TIMED_SCOPE(timerObj, "calculate gain");
                            auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                    float_type lambda) -> float_type {
                                    if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda)
                                                -(father.g * father.g) / (father.h + lambda);
                                    else
                                    return 0;
                            };

                            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
                            GHPair *gh_prefix_sum_data = hist.device_data();
                            float_type *gain_data = gain.device_data();
                            const auto missing_gh_data = missing_gh.device_data();
                            auto ignored_set_data = ignored_set.device_data();
                            //for lambda expression
                            float_type mcw = param.min_child_weight;
                            float_type l = param.lambda;
                            device_loop(n_bins, [=]__device__(int i) {
                                int nid0 = i / n_bins;
                                int nid = nid0 + nid_offset;
                                int fid = hist_fid[i % n_bins];
                                if (nodes_data[nid].is_valid && !ignored_set_data[fid]) {
                                    int pid = nid0 * n_column + hist_fid[i];
                                    GHPair father_gh = nodes_data[nid].sum_gh_pair;
                                    GHPair p_missing_gh = missing_gh_data[pid];
                                    GHPair rch_gh = gh_prefix_sum_data[i];
                                    float_type default_to_left_gain = max(0.f,
                                                                          compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                                    rch_gh = rch_gh + p_missing_gh;
                                    float_type default_to_right_gain = max(0.f,
                                                                           compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                                    if (default_to_left_gain > default_to_right_gain)
                                        gain_data[i] = default_to_left_gain;
                                    else
                                        gain_data[i] = -default_to_right_gain;//negative means default split to right

                                } else gain_data[i] = 0;
                            });
                            LOG(DEBUG) << "gain = " << gain;
                        }

                        SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                        {
                //            TIMED_SCOPE(timerObj, "get best gain");
                            auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                                if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                                    return get<0>(a) < get<0>(b) ? a : b;
                                else
                                    return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
                            };

                            auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

                            reduce_by_key(
                                    cuda::par,
                                    nid_iterator, nid_iterator + n_split,
                                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                                    make_discard_iterator(),
                                    best_idx_gain.device_data(),
                                    thrust::equal_to<int>(),
                                    arg_abs_max
                            );
                            LOG(DEBUG) << n_split;
                            LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
                        }

                        //get split points
                        {
                            const int_float *best_idx_gain_data = best_idx_gain.device_data();
                            auto hist_data = hist.device_data();
                            const auto missing_gh_data = missing_gh.device_data();
                            auto cut_val_data = cut.cut_points_val.device_data();

                            sp.resize(n_nodes_in_level);
                            auto sp_data = sp.device_data();
                            auto nodes_data = tree.nodes.device_data();

                            int column_offset = columns.column_offset;

                            auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                            device_loop(n_nodes_in_level, [=]__device__(int i) {
                                int_float bst = best_idx_gain_data[i];
                                float_type best_split_gain = get<1>(bst);
                                int split_index = get<0>(bst);
                                if (!nodes_data[i + nid_offset].is_valid) {
                                    sp_data[i].split_fea_id = -1;
                                    sp_data[i].nid = -1;
                                    return;
                                }
                                int fid = hist_fid[split_index];
                                sp_data[i].split_fea_id = fid + column_offset;
                                sp_data[i].nid = i + nid_offset;
                                sp_data[i].gain = fabsf(best_split_gain);
                                sp_data[i].fval = cut_val_data[split_index % n_bins];
                                sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
                                sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
                                sp_data[i].default_right = best_split_gain < 0;
                                sp_data[i].rch_sum_gh = hist_data[split_index];
                            });
                        }
                        
                    } else {
                        //otherwise
                        auto t_dp_begin = timer.now();
                        SyncArray<int> node_idx(n_instances);
                        SyncArray<int> node_ptr(n_nodes_in_level + 1);
                        {
                            TIMED_SCOPE(timerObj, "data partitioning");
                            SyncArray<int> nid4sort(n_instances);
                            nid4sort.copy_from(ins2node_id[device_id]);
                            sequence(cuda::par, node_idx.device_data(), node_idx.device_end(), 0);

                            cub_sort_by_key(nid4sort, node_idx);
                            auto counting_iter = make_counting_iterator < int > (nid_offset);
                            node_ptr.host_data()[0] =
                                    lower_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), nid_offset) -
                                    nid4sort.device_data();

                            upper_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), counting_iter,
                                        counting_iter + n_nodes_in_level, node_ptr.device_data() + 1);
                            LOG(DEBUG) << "node ptr = " << node_ptr;
                            cudaDeviceSynchronize();
                        }
                        auto t_dp_end = timer.now();
                        std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
                        this->total_dp_time += dp_used_time.count();


                        auto node_ptr_data = node_ptr.host_data();
                        auto node_idx_data = node_idx.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        //auto dense_bin_id_data = dense_bin_id.device_data();
                        auto max_num_bin = param.max_num_bin;

                        //new varibales
                        size_t last_hist_len = n_max_nodes/2;
                        size_t half_last_hist_len = n_max_nodes/4;

                        SyncArray<float_type> gain(len_hist);
                        SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                        sp.resize(n_nodes_in_level);

                        //test
                        // SyncArray<GHPair> test(2*n_bins);
                        // auto test_data = test.device_data();
                        for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                            size_t tmp_index = i;
                            int nid0_to_compute = i * 2;
                            int nid0_to_substract = i * 2 + 1;
                            int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                            int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                            if (max(n_ins_left, n_ins_right) == 0) 
                            {   
                                auto nodes_data = tree.nodes.device_data();
                                auto sp_data = sp.device_data();
                                device_loop(2, [=]__device__(int i) {
                                    if (!nodes_data[i + nid_offset+2*tmp_index].is_valid) {
                                        sp_data[i+2*tmp_index].split_fea_id = -1;
                                        sp_data[i+2*tmp_index].nid = -1;
                                    }
                                });
                                continue;
                            }
                            if (n_ins_left > n_ins_right)
                                swap(nid0_to_compute, nid0_to_substract);

                            size_t computed_hist_pos = nid0_to_compute%2;
                            size_t to_compute_hist_pos = 1-computed_hist_pos;

                            //compute
                            {
                                int nid0 = nid0_to_compute;
                                auto idx_begin = node_ptr.host_data()[nid0];
                                auto idx_end = node_ptr.host_data()[nid0 + 1];
                                auto hist_data = hist.device_data() + computed_hist_pos*n_bins;
                                this->total_hist_num++;

                                //reset zero
                                cudaMemset(hist_data, 0, n_bins*sizeof(GHPair));

                                    //new csr loop
                                    device_loop_hist_csr_node((idx_end - idx_begin),csr_row_ptr_data, [=]__device__(int i,int current_pos,int stride){
                                        //iid
                                        int iid = node_idx_data[i+idx_begin];
                                        int begin = csr_row_ptr_data[iid];
                                        int end = csr_row_ptr_data[iid+1];

                                        for(int j = begin+current_pos;j<end;j+=stride){
                                            //int fid = csr_col_idx_data[j];
                                            int bid = (int)csr_bin_id_data[j];

                                            //int feature_offset = cut_row_ptr_data[fid];
                                            const GHPair src = gh_data[iid];
                                            //GHPair &dest = hist_data[feature_offset + bid];
                                            GHPair &dest = hist_data[bid];

                                            if(src.h!= 0){
                                                atomicAdd(&dest.h, src.h);
                                            }

                                            if(src.g!= 0){
                                                atomicAdd(&dest.g, src.g);
                                            }
                                        
                                        }
                                    },n_block);

                                    //check result
                                    // check_hist_res(hist.host_data() + nid0 * n_bins,test.host_data()+computed_hist_pos*n_bins,n_bins);
                                
                            }

                            //subtract
                            auto t_copy_start = timer.now();
                            {
                                auto hist_data_computed = hist.device_data() + computed_hist_pos * n_bins;
                                auto hist_data_to_compute = hist.device_data() + to_compute_hist_pos * n_bins;
                                auto father_hist_data = last_hist.device_data() + (nid0_to_substract / 2) * n_bins;
                                
                                if(level%2==0){
                                    size_t st_pos = (((half_last_hist_len+(nid0_to_substract / 2)))%last_hist_len)* n_bins;
                                    father_hist_data = last_hist.device_data() + st_pos ;
                                }

                                device_loop(n_bins, [=]__device__(int i) {
                                    hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                                });
                            }
                            auto t_copy_end = timer.now();
                            std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                            this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);

                            //设计last_hist的拷贝策略

                            if(level<(param.depth-1)){
                                if(level%2==0){
                                    //even level
                                    // LOG(INFO)<<"start_pos in even level "<<2*i;
                                    cudaMemcpy(last_hist.device_data()+2*i*n_bins, hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                }
                                else{
                                    //odd level
                                    //start copy position
                                    size_t start_pos = ((half_last_hist_len+2*i)%last_hist_len)*n_bins;
                                    // LOG(INFO)<<"start_pos in odd level "<<((half_last_hist_len+2*i)%last_hist_len);
                                    cudaMemcpy(last_hist.device_data()+start_pos,hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                }
                            }


                            cudaDeviceSynchronize();

                            inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + 2*n_bins,
                                        hist.device_data(), 
                                        hist.device_data());


                            // check_hist_res(hist.host_data()+2*tmp_index*n_bins,test.host_data(),2*n_bins);

                            //copy to test array
                            // cudaMemcpy(test.device_data(),hist.device_data()+2*i*n_bins, 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);

                            { //missing 
                                auto nodes_data = tree.nodes.device_data();
                                auto missing_gh_data = missing_gh.device_data();
                                auto cut_row_ptr = cut.cut_row_ptr.device_data();
                                auto hist_data = hist.device_data();
                                int loop_len = n_column*2;
                                device_loop(loop_len, [=]__device__(int pid) {
                                    int nid0 = (pid / n_column);
                                    int nid = nid0 + nid_offset+2*tmp_index;
                                    if (!nodes_data[nid].splittable()) return;
                                    int fid = pid % n_column;
                                    if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
                                        GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
                                        missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
                                    }
                                });

                            }

                            {
                    //            TIMED_SCOPE(timerObj, "calculate gain");
                                auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                        float_type lambda) -> float_type {
                                        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                                        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda)
                                                -(father.g * father.g) / (father.h + lambda);
                                        else
                                        return 0;
                                };

                                const Tree::TreeNode *nodes_data = tree.nodes.device_data();
                                GHPair *gh_prefix_sum_data = hist.device_data();
                                float_type *gain_data = gain.device_data();
                                const auto missing_gh_data = missing_gh.device_data();
                                auto ignored_set_data = ignored_set.device_data();
                                //for lambda expression
                                float_type mcw = param.min_child_weight;
                                float_type l = param.lambda;
                                device_loop(2*n_bins, [=]__device__(int i) {
                                    int nid0 = i / n_bins;
                                    int nid = nid0 + nid_offset+2*tmp_index;
                                    int fid = hist_fid[i % n_bins];
                                    if (nodes_data[nid].is_valid && !ignored_set_data[fid]) {
                                        int pid = nid0 * n_column + hist_fid[i];
                                        GHPair father_gh = nodes_data[nid].sum_gh_pair;
                                        GHPair p_missing_gh = missing_gh_data[pid];
                                        GHPair rch_gh = gh_prefix_sum_data[i];
                                        float_type default_to_left_gain = max(0.f,
                                                                              compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                                        rch_gh = rch_gh + p_missing_gh;
                                        float_type default_to_right_gain = max(0.f,
                                                                               compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                                        if (default_to_left_gain > default_to_right_gain)
                                            gain_data[i] = default_to_left_gain;
                                        else
                                            gain_data[i] = -default_to_right_gain;//negative means default split to right

                                    } else gain_data[i] = 0;
                                });
                                LOG(DEBUG) << "gain = " << gain;
                            }

                            
                            {
                    //            TIMED_SCOPE(timerObj, "get best gain");
                                auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                                    if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                                        return get<0>(a) < get<0>(b) ? a : b;
                                    else
                                        return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
                                };

                                auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

                                reduce_by_key(
                                        cuda::par,
                                        nid_iterator, nid_iterator + 2*n_bins,
                                        make_zip_iterator(make_tuple(counting_iterator<int>(2*tmp_index*n_bins), gain.device_data())),
                                        make_discard_iterator(),
                                        best_idx_gain.device_data(),
                                        thrust::equal_to<int>(),
                                        arg_abs_max
                                );
                                LOG(DEBUG) << n_split;
                                LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
                            }

                            
                            //get split points
                            {
                                const int_float *best_idx_gain_data = best_idx_gain.device_data();
                                auto hist_data = hist.device_data();
                                const auto missing_gh_data = missing_gh.device_data();
                                auto cut_val_data = cut.cut_points_val.device_data();

                                
                                auto sp_data = sp.device_data();
                                auto nodes_data = tree.nodes.device_data();

                                int column_offset = columns.column_offset;

                                auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                                
                                device_loop(2, [=]__device__(int i) {
                                    int_float bst = best_idx_gain_data[i];
                                    float_type best_split_gain = get<1>(bst);
                                    int split_index = get<0>(bst);
                                    if (!nodes_data[i + nid_offset+2*tmp_index].is_valid) {
                                        sp_data[i+2*tmp_index].split_fea_id = -1;
                                        sp_data[i+2*tmp_index].nid = -1;
                                        return;
                                    }
                                    int fid = hist_fid[split_index];
                                    sp_data[i+2*tmp_index].split_fea_id = fid + column_offset;
                                    sp_data[i+2*tmp_index].nid = i + nid_offset+2*tmp_index;
                                    sp_data[i+2*tmp_index].gain = fabsf(best_split_gain);
                                    sp_data[i+2*tmp_index].fval = cut_val_data[split_index % n_bins];
                                    sp_data[i+2*tmp_index].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
                                    sp_data[i+2*tmp_index].fea_missing_gh = missing_gh_data[(i) * n_column + hist_fid[split_index]];
                                    sp_data[i+2*tmp_index].default_right = best_split_gain < 0;
                                    sp_data[i+2*tmp_index].rch_sum_gh = hist_data[i*n_bins+split_index%n_bins];

                                });
                                
                            }

                            


                        }  // end for each node

                        //clear array
                        //hist.resize(0);
                        //missing_gh.resize(0);
                        //gain.resize(0);
                        //best_idx_gain.resize(0);
                        
                            
                    }//end # node > 1
                    
                }
                
            }
            TEND(sort_time)
            total_sort_time_hist+=TINT(sort_time);
        }
        //calculate gain of each split

        
        
    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

//void HistTreeBuilder::update_ins2node_id() {
//    TIMED_FUNC(timerObj);
//    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
//        SyncArray<bool> has_splittable(1);
//        auto &columns = shards[device_id].columns;
//        //set new node id for each instance
//        {
////        TIMED_SCOPE(timerObj, "get new node id");
//            auto nid_data = ins2node_id[device_id].device_data();
//            const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
//            has_splittable.host_data()[0] = false;
//            bool *h_s_data = has_splittable.device_data();
//            int column_offset = columns.column_offset;
//
//            int n_column = columns.n_column;
//            auto dense_bin_id_data = dense_bin_id[device_id].device_data();
//            int max_num_bin = param.max_num_bin;
//            device_loop(n_instances, [=]__device__(int iid) {
//                int nid = nid_data[iid];
//                const Tree::TreeNode &node = nodes_data[nid];
//                int split_fid = node.split_feature_id;
//                if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
//                    h_s_data[0] = true;
//                    unsigned char split_bid = node.split_bid;
//                    unsigned char bid = dense_bin_id_data[(long long)iid * (long long)n_column + (long long)split_fid - (long long)column_offset];
//                    bool to_left = true;
//                    if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
//                        to_left = false;
//                    if (to_left) {
//                        //goes to left child
//                        nid_data[iid] = node.lch_index;
//                    } else {
//                        //right child
//                        nid_data[iid] = node.rch_index;
//                    }
//                }
//            });
//        }
//        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
//        has_split[device_id] = has_splittable.host_data()[0];
//    });
//}

void HistTreeBuilder::init(const DataSet &dataset, const GBMParam &param) {
    TreeBuilder::init(dataset, param);
    //TODO refactor
    //init shards
    int n_device = param.n_device;
    shards = vector<Shard>(n_device);
    vector<std::unique_ptr<SparseColumns>> v_columns(param.n_device);
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].reset(&shards[i].columns);
        shards[i].ignored_set = SyncArray<bool>(dataset.n_features());
    }
    SparseColumns columns;
    if(dataset.use_cpu)
        columns.csr2csc_cpu(dataset, v_columns);
    else
        columns.csr2csc_gpu(dataset, v_columns);
    cut = vector<HistCut>(param.n_device);
    //dense_bin_id = MSyncArray<unsigned char>(param.n_device);
    last_hist = MSyncArray<GHPair>(param.n_device);

    //csr bin id
    csr_bin_id = MSyncArray<int>(param.n_device);
    csr_row_ptr = MSyncArray<int>(param.n_device);
    csr_col_idx = MSyncArray<int>(param.n_device);

    //bin_id_origin = MSyncArray<unsigned char>(param.n_device);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        if(dataset.use_cpu)
            cut[device_id].get_cut_points2(shards[device_id].columns, param.max_num_bin, n_instances);
        else
            cut[device_id].get_cut_points3(shards[device_id].columns, param.max_num_bin, n_instances);
        last_hist[device_id].resize((1 << (param.depth-2)) * cut[device_id].cut_points_val.size());
        LOG(INFO)<<"last hist size is "<<((1 << (param.depth-2)) * cut[device_id].cut_points_val.size())*8/1e9;
        
        //set data
        auto y_predict_data = y_predict[device_id].device_data();
        device_loop(y_predict[device_id].size(), [=]__device__(size_t i) {
            y_predict_data[i] = -0.975106f;
        });
   });
    get_bin_ids();
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].release();
    }
    SyncMem::clear_cache();
    int gpu_num;
    cudaError_t err = cudaGetDeviceCount(&gpu_num);
    std::atexit([](){
        SyncMem::clear_cache();
    });
}

//new func for ins2node update
//new ins2node update
void HistTreeBuilder::update_ins2node_id() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        
        auto &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        
        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];
        
        auto csr_row_ptr_data = csr_row_ptr.device_data();
        auto csr_bin_id_data = csr_bin_id.device_data();

        auto cut_row_ptr = cut.cut_row_ptr.device_data();

        using namespace thrust;
        int n_column = columns.n_column;
        //int nnz = columns.nnz;
        
        // int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);

        SyncArray<bool> has_splittable(1);
        auto nid_data = ins2node_id[device_id].device_data();
        const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.device_data();
        int column_offset = columns.column_offset;
        //auto max_num_bin = param.max_num_bin;

        //两种思路
        //1.得到划分的split bin_id，然后从csr_bin_id中寻找，这样寻找的范围是一个instance中nnz的长度
        //2、得到划分的split feature 和split feature，在csr中寻找instance的feature和对应的value，这需要保留value数组
        //选择第一种
        auto binary_search = [=]__device__( size_t search_begin,  size_t search_end, 
                                            const size_t cut_begin, const size_t cut_end,
                                            const int *csr_bin_id_data) {
            int previous_middle = -1;
            while (search_begin != search_end) {
                int middle = search_begin + (search_end - search_begin)/2;

                if(middle == previous_middle){
                    break;
                }
                previous_middle = middle;
                auto tmp_bin_id = csr_bin_id_data[middle];

                if(tmp_bin_id >= cut_begin && tmp_bin_id < cut_end){
                    return tmp_bin_id;
                }
                else if (tmp_bin_id < cut_begin){
                    search_begin = middle;
                }
                else{
                    search_end = middle;
                }
            }
            //missing values
            return -1;
        };

        auto loop_search = [=]__device__( size_t search_begin,  size_t search_end, 
                                            const size_t cut_begin, const size_t cut_end,
                                            const int *csr_bin_id_data) {
            for(int i =search_begin;i<=search_end;i++){
                auto bin_id = csr_bin_id_data[i];
                if(bin_id >= cut_begin && bin_id <cut_end){
                    return bin_id;
                }
            }
            return -1;
        };
        //update instance to node map
        device_loop(n_instances, [=]__device__(int iid) {
            int nid = nid_data[iid];
            const Tree::TreeNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                int split_bid = (int)node.split_bid+cut_row_ptr[split_fid]; 
                int bid = binary_search(csr_row_ptr_data[iid],csr_row_ptr_data[iid+1],
                                        cut_row_ptr[split_fid],cut_row_ptr[split_fid+1],
                                        csr_bin_id_data);
                bool to_left = true;
                if ((bid == -1 && node.default_right) || (bid <= split_bid && bid>=0))
                    to_left = false;
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                }
            }
        });
        
       
        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
        has_split[device_id] = has_splittable.host_data()[0];
    });
}
//new update func with level
void HistTreeBuilder::update_ins2node_id(int level) {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        
        int n_nodes_in_level = static_cast<int>(pow(2, level));
        //当前level节点的开始index
        int nid_offset = static_cast<int>(pow(2, level) - 1);
        
        auto &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        auto &dense_bin_id = this->dense_bin_id[device_id];
        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_col_idx = this->csr_col_idx[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];

        auto csr_row_ptr_data = csr_row_ptr.device_data();
        auto csr_col_idx_data = csr_col_idx.device_data();
        auto csr_bin_id_data = csr_bin_id.device_data();

        using namespace thrust;
        int n_column = columns.n_column;
        // int nnz = columns.nnz;
        
        // int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);
       

        int using_col_num = 0;
        //first loop for splitable
        //for(int i =0;i<n_nodes_in_level;++i){
        //    const Tree::TreeNode &node = trees[device_id].nodes.host_data()[i+nid_offset];

        //    //mat same fid ??
        //    if(node.splittable()){
        //        using_col_num++;
        //    }
        //}
        //
        //if(using_col_num==0){
        //   has_split[device_id] = false;
        //   return;
        //}

        SyncArray<int> col_idx2feature_map(32);
        SyncArray<int> feature2col_idx_map(n_column);
        auto col_idx2feature_map_host = col_idx2feature_map.host_data();
        auto feature2col_idx_map_host = feature2col_idx_map.host_data();
        
        int tmp = 0;
        for(int i=0;i<n_nodes_in_level;++i){
            
            //current split node
            const Tree::TreeNode &node = trees[device_id].nodes.host_data()[i+nid_offset];
            if(node.splittable()){ 
                int split_fid = node.split_feature_id;
                col_idx2feature_map_host[tmp] = split_fid;
                feature2col_idx_map_host[split_fid] = tmp;
                tmp++;
            }
        }
        using_col_num = tmp;
        if(using_col_num==0){
           has_split[device_id] = false;
           return;
        }
        SyncArray<bool> has_splittable(1);
        auto nid_data = ins2node_id[device_id].device_data();
        const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.device_data();
        int column_offset = columns.column_offset;
        auto max_num_bin = param.max_num_bin;
        auto dense_bin_id_data = dense_bin_id.device_data();
        

        auto col_idx2feature_map_device = col_idx2feature_map.device_data();
        auto feature2col_idx_map_device = feature2col_idx_map.device_data();
        auto &bin_id_origin = this->bin_id_origin[device_id];
        auto bin_id_origin_data = bin_id_origin.device_data();
        auto csc_row_idx_data = columns.csc_row_idx_origin.device_data();
        //size_t num_block = (n_instances/n_column)/256+1;
        size_t num_block = (n_instances)/256+1;
        //update process
        device_loop_part_dense_bin_id_csc(using_col_num,columns.csc_col_ptr_origin.device_data(),col_idx2feature_map_device,[=]__device__(int col_idx,int i){

            int ins_idx = csc_row_idx_data[i];
            auto bid = (unsigned char)bin_id_origin_data[i];
            size_t pos = (unsigned int)ins_idx*using_col_num+col_idx;

            dense_bin_id_data[pos] = bid;
        },num_block);

        device_loop_part_update_node(n_instances, 0, [=]__device__(size_t idx, size_t tt ) {
 
            int iid  = idx;
            int nid = nid_data[iid];
            const Tree::TreeNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                unsigned char split_bid = node.split_bid;
                size_t pos = (unsigned int)iid* using_col_num + feature2col_idx_map_device[split_fid];
                unsigned char bid = dense_bin_id_data[pos];
                bool to_left = true;
                if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
                    to_left = false;
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                }
            }
        });
        device_loop_part_dense_bin_id_csc(using_col_num,columns.csc_col_ptr_origin.device_data(),col_idx2feature_map_device,[=]__device__(int col_idx,int i){

            int ins_idx = csc_row_idx_data[i];
            size_t pos = (unsigned int)ins_idx*using_col_num+col_idx;

            dense_bin_id_data[pos] = max_num_bin;

        },num_block);

        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
        has_split[device_id] = has_splittable.host_data()[0];
    });
}
