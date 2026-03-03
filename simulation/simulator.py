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

# cluster_name = ['Earth', 'Saturn', 'Uranus', 'Venus']
cluster_name = ['Earth', 'Saturn']

cluster_node_num = [784, 2072, 2072, 968]
cluster_num = len(cluster_name)

def main(args):
    code_start = time.perf_counter()

    """Logger Setting"""
    log_dir = f"{args.log_dir}/{args.experiment_name}/{args.global_policy}+{args.scheduler}" # './log/train/min_load_first+qssf'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = utils.logger_init(file=f"{log_dir}/cluster_num_{cluster_num}") #'./log/train/min_load_first+qssf'

    clusters = []  # 存储多个 Cluster 对象
    for i, cn in enumerate(cluster_name):
        node_num = cluster_node_num[i]
        cl = cluster.Cluster(cn, node_num, args.num_gpus_per_node, args.num_cpus_per_node)
        clusters.append(cl)

    trace_df, start_ts = utils.get_trace(args.experiment_name, args.trace_dir, read_full=False, cluster_name=cluster_name, cluster_num = cluster_num)
    logger.info(f"Total Job Number in Cluster Training: {len(trace_df)}")

    trace = utils.trace_parser(trace_df) #为每条trace添加额外参数

    # colocate_df = pd.read_csv("data_test/colocate_info.csv")
    # updater = ColocateUpdater(colocate_df)
    # estimator = CombinedEstimator(args)

    scheduler = GlobalScheduler(args.global_policy, trace, cluster_num, clusters, args.placer, log_dir, args.scheduler, logger, start_ts, None, None) # estimator, updater
    scheduler.run()

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
    )

    parser.add_argument("-s", "--scheduler", default="qssf", choices=utils.get_available_schedulers(), type=str, help="Scheduler Algorithm")
    parser.add_argument("-e", "--experiment-name", default="train", type=str, help="Experiment Name")
    parser.add_argument("-t", "--cl-dir", nargs='+', default="/home/hyn/Luicd_global/simulation/data/clusters", type=str, help="Trace File Directory")
    parser.add_argument("-tl", "--trace-dir", default="/home/hyn/Luicd_global/simulation/data", type=str, help="Trace File Directory")
    parser.add_argument("-l", "--log-dir", default="/home/hyn/Luicd_global/log", type=str, help="Log Directory")
    parser.add_argument("-p", "--placer", default="consolidate", choices=utils.get_available_placers(), type=str, help="Placer Algorithm")
    parser.add_argument("--num_gpus_per_node", type=int, default=8, help=("Number of GPUs per node"))
    parser.add_argument("--num_cpus_per_node", type=int, default=96, help=("Number of CPU cores per node"))

    # for lucid
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

    args = parser.parse_args()

    main(args)
