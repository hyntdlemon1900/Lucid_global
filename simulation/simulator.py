import time
import os
import argparse
import multiprocessing
import pandas as pd
import utils
from utils import GlobalScheduler
import cluster
from estimator import CombinedEstimator, PhillyEstimator
from updater import ColocateUpdater

CLUSTER_NUM = 3

os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())
def launch_scheduler(args_tuple):
        scheduler = GlobalScheduler(*args_tuple)
        return scheduler.run()

def main(args):
    code_start = time.perf_counter()
    if args.experiment_name == 'train':
        vc_num = utils.VC_NUM_Venus_train
        vc_name_list = utils.VC_NAME_LIST_Venus_train
    else:
        vc_num = utils.VC_NUM_Venus_eval
        vc_name_list = utils.VC_NAME_LIST_Venus_eval
    """Logger Setting"""
    log_dir = f"{args.log_dir}/{args.dataset}_{args.experiment_name}/{args.global_policy}_{args.scheduler}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = utils.logger_init(file=f"{log_dir}/cluster_num_{CLUSTER_NUM}")

    """Infrastructure & Trace Initialization"""
    all_vc_dicts = []  # 存放所有集群的配置
    for i in range(CLUSTER_NUM):
        vc_df = pd.read_csv(f"{args.cl_dir}{i+1}/vc_config_train.csv", index_col=0)
        vc_dict = vc_df.to_dict()["num"]
        all_vc_dicts.append(vc_dict)

    CLUSTERs = []  # 存储多个 Cluster 对象
    for vc_dict in all_vc_dicts:
        CLUSTER = cluster.Cluster(vc_dict, args.num_gpus_per_node, args.num_cpus_per_node)
        CLUSTERs.append(CLUSTER)

    trace_df, start_ts = utils.get_trace(args.experiment_name, args.trace_dir, read_full=True, cluster_num = CLUSTER_NUM)
    logger.info(f"Total Job Number in Cluster Training: {len(trace_df)}")

    trace = utils.trace_parser(trace_df) #为每条trace添加额外参数

    # if args.scheduler in utils.PROFILER_ENABLED_SCHEDULERS and not args.sweep:
    #     if args.profiler_auto:
    #         vc_dict, prof_scale, prof_time, prof_factor = utils.profiler_config(args.experiment_name, vc_dict)
    #         trace = utils.trace_profile(trace, prof_scale, prof_time, prof_factor, args.placer, log_dir, logger, start_ts)
    #     else:
    #         # NOTE: NOT update vc_dict for manual configuration
    #         prof_vc = utils.check_profiler_scale_available(
    #             args.experiment_name, args.profiler_scale, vc_dict, prof_locate_vc=None
    #         )
    #         trace = utils.trace_profile(
    #             trace, args.profiler_scale, args.profiler_time, args.profiler_factor, args.placer, log_dir, logger, start_ts
    #         )
    #         return
    #     logger.info(f"Profiling Execution Time: {round(time.perf_counter() - code_start, 2)}s")


    # colocate_df = pd.read_csv("data_test/colocate_info.csv")
    # updater = ColocateUpdater(colocate_df)
    # estimator = CombinedEstimator(args)


    if args.processes is None:
        process_num = min(vc_num, os.cpu_count())
        
    else:
        process_num = args.processes

    all_args_list = []
    for i in range(vc_num):
        vc_name = vc_name_list[i]
        all_vc_list = []
        for j, CLUSTER in enumerate(CLUSTERs):
            all_vc_list.append(CLUSTER.vc_list[i])
        all_args_list.append((args.global_policy, trace, vc_name, CLUSTER_NUM, all_vc_list, args.placer, log_dir, [args.scheduler, args.scheduler, args.scheduler], logger, start_ts, [None, None, None], [None, None, None])) # estimator, updater

    scheduler = GlobalScheduler(*all_args_list[12])
    scheduler.run()
    # with multiprocessing.Pool(processes=process_num) as p:
    #     results = [p.apply_async(launch_scheduler, (args_list,)) for args_list in all_args_list]
    #     results = [result.get() for result in results]

    # if args.sweep:
    #     for policy in utils.get_sweep_schedulers():
    #         utils.cluster_concatenate(policy, args.placer, log_dir, args.trace_dir)
    # else:
    #     utils.cluster_concatenate(args.scheduler, args.placer, log_dir, args.trace_dir)
    #     utils.cluster_analysis(args.placer, log_dir, args.trace_dir)

    #     """Fast query result"""
    #     sched_label = args.scheduler + "_consolidate"
    #     jct_df = pd.read_csv(f"{log_dir}/jct_avg_consolidate.csv", index_col=0)
    #     jct = jct_df.at["all", sched_label]
    #     que_df = pd.read_csv(f"{log_dir}/que_avg_consolidate.csv", index_col=0)
    #     que = que_df.at["all", sched_label]
    #     logger.info(f"Summary of {args.scheduler}: Avg. JCT {jct}s, Avg. Queue {que}s")

    logger.info(f"Execution Time: {round(time.perf_counter() - code_start, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator")

    parser.add_argument(
        "--global_policy",
        choices=[
            "round_robin",          # 轮询
            "random",               # 随机
            "max_free_gpus",        # 空闲 GPU 最多
            "min_load_first",       # 负载最轻 (used_gpus + queue_len)
            "best_fit",             # 满足需求且剩余 GPU 最少
            "power_of_choice"       # 先随机 2 个，再选负载轻的
        ],
        #default="random",
        default="min_load_first",
        help=(
            "全局调度策略：round_robin | random | max_free_gpus | min_load_first | best_fit | power_of_choice")
    )

    parser.add_argument("-s", "--scheduler", default="sjf", choices=utils.get_available_schedulers(), type=str, help="Scheduler Algorithm")
    parser.add_argument("-d", "--dataset", default="Venus", type=str, help="dataset")
    parser.add_argument("-e", "--experiment-name", default="train", type=str, help="Experiment Name")
    parser.add_argument("-t", "--cl-dir", nargs='+', default="./data_test/Venus_cl", type=str, help="Trace File Directory")
    parser.add_argument("-tl", "--trace-dir", default="./data_test", type=str, help="Trace File Directory")
    parser.add_argument("-l", "--log-dir", default="./log", type=str, help="Log Directory")

    parser.add_argument("-p", "--placer", default="consolidate", choices=utils.get_available_placers(), type=str, help="Placer Algorithm")

    parser.add_argument("--profiler-auto", default=1, type=int, help="Use default profiling setting, disable below (time, scale, factor).")
    # For ablation study
    parser.add_argument("--profiler-time", default=500, type=int, help="Time limit in profiler, unit: second")
    parser.add_argument("--profiler-scale", default=6, type=int, help="Number of nodes applied in profiler")
    parser.add_argument("--profiler-factor", default=6, type=int, help="Maximum GPU number to be profiled = factor x scale")

    parser.add_argument("--colocate", default=0, type=int, help="Whether to enable GPU sharing")
    parser.add_argument("--pollux-idx", default=None, type=int, help="Index of Pollux Trace")
    parser.add_argument("--sweep", action="store_true", default=False, help="Run All Scheduler Policies in One Time")
    parser.add_argument("-j", "--processes", type=int, default=None, help=("Number of processes to use in multiprocessing.Pool" "(use as many as available if not specified)"))
    parser.add_argument("--timeout", default=1209600, type=int, help="Timeout (in seconds), default 14 days")
    parser.add_argument("--num_gpus_per_node", type=int, default=8, help=("Number of GPUs per node"))
    parser.add_argument("--num_cpus_per_node", type=int, default=96, help=("Number of CPU cores per node"))

    args = parser.parse_args()

    main(args)
