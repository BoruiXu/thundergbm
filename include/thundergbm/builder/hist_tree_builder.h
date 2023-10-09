//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_HIST_TREE_BUILDER_H
#define THUNDERGBM_HIST_TREE_BUILDER_H

#include <thundergbm/hist_cut.h>
#include "thundergbm/common.h"
#include "shard.h"
#include "tree_builder.h"


class HistTreeBuilder : public TreeBuilder {
public:

    void init(const DataSet &dataset, const GBMParam &param) override;

    void get_bin_ids();

    void find_split(int level, int device_id) override;

    virtual ~HistTreeBuilder(){};

    void update_ins2node_id() override;

    //new func
    void update_ins2node_id(int level) override;


private:
    vector<HistCut> cut;
    MSyncArray<unsigned char> char_dense_bin_id;
    MSyncArray<unsigned char> dense_bin_id;
    MSyncArray<GHPair> last_hist;

    //store csr dense_bin_id
    MSyncArray<float_type> csr_bin_id;
    MSyncArray<float_type> bin_id_origin; //csc format
    MSyncArray<int> csr_row_ptr;
    MSyncArray<int> csr_col_idx;

    double build_hist_used_time=0;
    int build_n_hist = 0;
    int total_hist_num = 0;
    double total_dp_time = 0;
    double total_copy_time = 0;
    bool use_gpu = 1;

};


#endif //THUNDERGBM_HIST_TREE_BUILDER_H
