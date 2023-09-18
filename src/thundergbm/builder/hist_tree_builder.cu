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
    cusparseCsr2cscEx2(handle, n_feature, n_instance, nnz, 
                            csc_val.device_data(),csc_col_ptr.device_data(),csc_row_idx.device_data(),
                            val.device_data(),row_ptr.device_data(),col_idx.device_data(),
                            data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, tmp_buffer.device_data());

    cudaDeviceSynchronize();
    LOG(INFO)<<"return status is "<<status;
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    //csc_val.resize(0);
    //csc_col_ptr.resize(0);
    //csc_row_idx.resize(0);


}
void HistTreeBuilder::get_bin_ids() {
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        SparseColumns &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        auto &dense_bin_id = this->dense_bin_id[device_id];
        using namespace thrust;
        int n_column = columns.n_column;
        int nnz = columns.nnz;
        auto cut_row_ptr = cut.cut_row_ptr.device_data();
        auto cut_points_ptr = cut.cut_points_val.device_data();
        //auto csc_val_data = columns.csc_val.device_data();
        SyncArray<unsigned char> bin_id;
        bin_id.resize(columns.nnz);
        auto bin_id_data = bin_id.device_data();
        int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);

        //original order csc
        auto csc_val_origin_data = columns.csc_val_origin.device_data();
        //SyncArray<float> bin_id_origin;
        auto &bin_id_origin = this->bin_id_origin[device_id];
        bin_id_origin.resize(columns.nnz);
        auto bin_id_origin_data = bin_id_origin.device_data();

        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_col_idx = this->csr_col_idx[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];
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
            //device_loop_2d(n_column, columns.csc_col_ptr.device_data(), [=]__device__(int cid, int i) {
            //    auto search_begin = cut_points_ptr + cut_row_ptr[cid];
            //    auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
            //    auto val = csc_val_data[i];
            //    bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            //}, n_block);
            //for original order csc
            device_loop_2d(n_column, columns.csc_col_ptr_origin.device_data(), [=]__device__(int cid, int i) {
                auto search_begin = cut_points_ptr + cut_row_ptr[cid];
                auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
                auto val = csc_val_origin_data[i];
                bin_id_origin_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            }, n_block);

        }
        //csc2csr
        csc2csr(columns.csc_col_ptr_origin,columns.csc_row_idx_origin,bin_id_origin,
                csr_row_ptr,csr_col_idx,csr_bin_id,
                n_instances,n_column);

        auto max_num_bin = param.max_num_bin;
        
        //long long total_size = (long long)20*(long long)1024*(long long)1024*(long long)1024;//10GB

        //size_t row_part_size = total_size/n_column;
        ////int loop_num = n_instances/row_part_size+1;
        //size_t current_dense_size = (row_part_size)*(long long)n_column;
        //
        //dense_bin_id.resize(current_dense_size);
        //auto dense_bin_id_data = dense_bin_id.device_data();
        //
        //device_loop(current_dense_size, [=]__device__(size_t i) {
        //    dense_bin_id_data[i] = max_num_bin;
        //});
        
        //here 32 is the max nodes in one level
        size_t current_dense_size = (long long)n_instances*32;

        dense_bin_id.resize(current_dense_size);
        auto dense_bin_id_data = dense_bin_id.device_data();
        
        device_loop(current_dense_size, [=]__device__(size_t i) {
            dense_bin_id_data[i] = max_num_bin;
        });

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
    auto &dense_bin_id = this->dense_bin_id[device_id];
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
    //find the best split locally
    {
        using namespace thrust;
        auto t_build_start = timer.now();

        //calculate split information for each split
        SyncArray<GHPair> hist(n_max_splits);
        SyncArray<GHPair> missing_gh(n_partition);
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
                        
                            int fid = csr_col_idx_data[j];
                            int bid = (int)csr_bin_id_data[j];
                            
                            int feature_offset = cut_row_ptr_data[fid];
                            const GHPair src = gh_data[i];
                            //GHPair &dest = hist_test_data[feature_offset + bid]; 
                            GHPair &dest = hist_data[feature_offset + bid]; 
                            if(src.h != 0)
                                atomicAdd(&dest.h, src.h);
                            if(src.g != 0)
                                atomicAdd(&dest.g, src.g);                            
                        
                        },n_block);
                        
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
                        for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                            int nid0_to_compute = i * 2;
                            int nid0_to_substract = i * 2 + 1;
                            int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                            int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                            if (max(n_ins_left, n_ins_right) == 0) continue;
                            if (n_ins_left > n_ins_right)
                                swap(nid0_to_compute, nid0_to_substract);

                            //compute
                            {
                                int nid0 = nid0_to_compute;
                                auto idx_begin = node_ptr.host_data()[nid0];
                                auto idx_end = node_ptr.host_data()[nid0 + 1];
                                auto hist_data = hist.device_data() + nid0 * n_bins;
                                this->total_hist_num++;

                                    //new csr loop

                                    //SyncArray<GHPair> hist_test(n_bins);
                                    //auto hist_test_data = hist_test.device_data();
                                    device_loop_hist_csr_node((idx_end - idx_begin),csr_row_ptr_data, [=]__device__(int i,int current_pos,int stride){
                                        //iid
                                        int iid = node_idx_data[i+idx_begin];
                                        int begin = csr_row_ptr_data[iid];
                                        int end = csr_row_ptr_data[iid+1];

                                        for(int j = begin+current_pos;j<end;j+=stride){
                                            int fid = csr_col_idx_data[j];
                                            int bid = (int)csr_bin_id_data[j];

                                            int feature_offset = cut_row_ptr_data[fid];
                                            const GHPair src = gh_data[iid];
                                            GHPair &dest = hist_data[feature_offset + bid];
                                            //GHPair &dest = hist_test_data[feature_offset + bid];

                                            if(src.h!= 0){
                                                atomicAdd(&dest.h, src.h);
                                            }

                                            if(src.g!= 0){
                                                atomicAdd(&dest.g, src.g);
                                            }
                                        
                                        }
                                    },n_block);

                                    //check result
                                    //check_hist_res(hist.host_data() + nid0 * n_bins,hist_test.host_data(),n_bins);
                                
                            }

                            //subtract
                            auto t_copy_start = timer.now();
                            {
                                auto hist_data_computed = hist.device_data() + nid0_to_compute * n_bins;
                                auto hist_data_to_compute = hist.device_data() + nid0_to_substract * n_bins;
                                auto father_hist_data = last_hist.device_data() + (nid0_to_substract / 2) * n_bins;
                                device_loop(n_bins, [=]__device__(int i) {
                                    hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                                });
                            }
                            auto t_copy_end = timer.now();
                            std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                            this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
                        }  // end for each node
                    }//end # node > 1
                    last_hist.copy_from(hist);
                    cudaDeviceSynchronize();
                }
                LOG(DEBUG) << "level: " << level;
                LOG(DEBUG) << "hist new = " << hist;
                auto t_build_hist_end = timer.now();
                std::chrono::duration<double> bh_used_time = t_build_hist_end - t_build_start;
                this->build_hist_used_time += bh_used_time.count();
                this->build_n_hist++;
                LOG(DEBUG) << "-------------->>> build_hist_used_time: " << bh_used_time.count();
                LOG(DEBUG) << "-------------->>> build_num_hist: " << this->build_n_hist;
                LOG(DEBUG) << "-------------->>> total_build_hist_used_time: " << this->build_hist_used_time - this->total_dp_time;
                LOG(DEBUG) << "-------------->>> n_hist::::: " << this->total_hist_num;
                LOG(DEBUG) << "-------------->>> dp_time::::: " << this->total_dp_time;
                LOG(DEBUG) << "-------------->>> cp_time::::: " << this->total_copy_time;

                //LOG(DEBUG) << "cutfid = " << cut.cut_fid;
                inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + n_split,
                                      hist.device_data(), hist.device_data());
                LOG(DEBUG) << hist;

                auto nodes_data = tree.nodes.device_data();
                auto missing_gh_data = missing_gh.device_data();
                auto cut_row_ptr = cut.cut_row_ptr.device_data();
                auto hist_data = hist.device_data();
                device_loop(n_partition, [=]__device__(int pid) {
                    int nid0 = pid / n_column;
                    int nid = nid0 + nid_offset;
                    if (!nodes_data[nid].splittable()) return;
                    int fid = pid % n_column;
                    if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
                        GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
                        missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
                    }
                });
                LOG(DEBUG) << missing_gh;
            }
            TEND(sort_time)
            total_sort_time_hist+=TINT(sort_time);
        }
        //calculate gain of each split
        SyncArray<float_type> gain(n_max_splits);
        {
//            TIMED_SCOPE(timerObj, "calculate gain");
            auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                    float_type lambda) -> float_type {
                    if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
            (father.g * father.g) / (father.h + lambda);
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
            device_loop(n_split, [=]__device__(int i) {
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
    dense_bin_id = MSyncArray<unsigned char>(param.n_device);
    last_hist = MSyncArray<GHPair>(param.n_device);

    //csr bin id
    csr_bin_id = MSyncArray<float_type>(param.n_device);
    csr_row_ptr = MSyncArray<int>(param.n_device);
    csr_col_idx = MSyncArray<int>(param.n_device);

    bin_id_origin = MSyncArray<float_type>(param.n_device);

    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        if(dataset.use_cpu)
            cut[device_id].get_cut_points2(shards[device_id].columns, param.max_num_bin, n_instances);
        else
            cut[device_id].get_cut_points3(shards[device_id].columns, param.max_num_bin, n_instances);
        last_hist[device_id].resize((1 << (param.depth-1)) * cut[device_id].cut_points_val.size());
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

        SyncArray<bool> has_splittable(1);
        auto nid_data = ins2node_id[device_id].device_data();
        const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.device_data();
        int column_offset = columns.column_offset;
        auto max_num_bin = param.max_num_bin;

        //int loop_num = 20;//1000;
        //size_t row_part_size = n_instances/loop_num;
        long long total_size = (long long)20*(long long)1024*(long long)1024*(long long)1024;//10GB

        size_t row_part_size = total_size/n_column;
        int loop_num = n_instances/row_part_size+1;
        //size_t current_dense_size = (row_part_size)*(long long)n_column;
        
        //initialize dense bin id
        size_t current_row_size = row_part_size;
        
        auto dense_bin_id_data = dense_bin_id.device_data();

        
        for(int l=0;l<loop_num;l++){

            
            //last one 
            if(l==loop_num-1){
                current_row_size = (n_instances-(loop_num-1)*row_part_size);
            }
            int start_row = l * row_part_size;
            

            //generate dense bin id 
            //this row is a map value not the real row index
            device_loop_part_dense_bin_id(current_row_size, csr_row_ptr_data, start_row,[=]__device__(int row, int i) {
                
                int fid = csr_col_idx_data[i];
                auto bid = (unsigned char)csr_bin_id_data[i];
                size_t pos = (long long)row * (long long)n_column + (long long)fid;
                dense_bin_id_data[pos] = bid;

            });
            
            //update ins2node information
            //set new node id for each instance
            device_loop_part_update_node(current_row_size, start_row, [=]__device__(size_t idx, size_t start_row) {

                int iid  = idx+start_row;
                int nid = nid_data[iid];
                const Tree::TreeNode &node = nodes_data[nid];
                int split_fid = node.split_feature_id;
                if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                    h_s_data[0] = true;
                    unsigned char split_bid = node.split_bid;
                    size_t pos = idx * n_column + split_fid - column_offset;
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
            
            //recover max_bin_id
            device_loop_part_dense_bin_id(current_row_size, csr_row_ptr_data, start_row,[=]__device__(int row, int i) {
                
                int fid = csr_col_idx_data[i];
                size_t pos = (long long)row * (long long)n_column + (long long)fid;
                dense_bin_id_data[pos] = max_num_bin;

            });


        }

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
        for(int i =0;i<n_nodes_in_level;++i){
            const Tree::TreeNode &node = trees[device_id].nodes.host_data()[i+nid_offset];

            //mat same fid ??
            if(node.splittable()){
                using_col_num++;
            }
        }


        SyncArray<int> col_idx2feature_map(using_col_num);
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
        //update process
        device_loop_part_dense_bin_id_csc(using_col_num,columns.csc_col_ptr_origin.device_data(),col_idx2feature_map_device,[=]__device__(int col_idx,int i){

            int ins_idx = csc_row_idx_data[i];
            auto bid = (unsigned char)bin_id_origin_data[i];
            size_t pos = ins_idx*using_col_num+col_idx;

            dense_bin_id_data[pos] = bid;

        },n_instances/256);

        device_loop_part_update_node(n_instances, 0, [=]__device__(size_t idx, size_t tt ) {
 
            int iid  = idx;
            int nid = nid_data[iid];
            const Tree::TreeNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                unsigned char split_bid = node.split_bid;
                size_t pos = iid* using_col_num + feature2col_idx_map_device[split_fid];
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
            size_t pos = ins_idx*using_col_num+col_idx;

            dense_bin_id_data[pos] = max_num_bin;

        },n_instances/256);

        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
        has_split[device_id] = has_splittable.host_data()[0];
    });
}
