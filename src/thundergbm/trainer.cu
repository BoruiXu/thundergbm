//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include "cuda_runtime_api.h"

#include <thundergbm/tree.h>
#include <thundergbm/trainer.h>
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"
#include "time.h"
#include "thundergbm/booster.h"
#include "chrono"
#include <thundergbm/parser.h>
using namespace std;
long long total_hist_time = 0;
long long total_split_update_time = 0;
long long total_evaluate_time = 0;
long long total_exact_prefix_sum_time = 0;


vector<vector<Tree>> TreeTrainer::train(GBMParam &param, const DataSet &dataset) {
    if (param.tree_method == "auto")
        if (dataset.n_features() > 20000)
            param.tree_method = "exact";
        else
            param.tree_method = "hist";

    //correct the number of classes
    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if(param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }

    vector<vector<Tree>> boosted_model;
    Booster booster;
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    booster.init(dataset, param);
    auto stop_init = timer.now();
    std::chrono::duration<float> init_time = stop_init - start;
    for (int i = 0; i < param.n_trees; ++i) {
        //one iteration may produce multiple trees, depending on objectives
        booster.boost(boosted_model,i+1,param.n_trees);
    }
    LOG(INFO)<<"total  hist construction time is "<<total_hist_time/1e6;
    //LOG(INFO)<<"total  exact prefix sum time is "<<total_exact_prefix_sum_time/1e6;
    LOG(INFO)<<"total evaluate time is "<<total_evaluate_time/1e6;
    LOG(INFO)<<"total split and update is "<<total_split_update_time/1e6;
    LOG(INFO)<<"initialization time = "<<init_time.count();
    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO)<<"other time is "<<training_time.count()-(total_hist_time+total_evaluate_time+total_split_update_time)/1e6;
    LOG(INFO) << "all training time = " << training_time.count();
    //LOG(INFO)<<"total time = "<<convert_time+training_time.count();

    std::atexit([]() {
        SyncMem::clear_cache();
    });
	// SyncMem::clear_cache();
	return boosted_model;
}
