import profile
import sys
import os
import logging
import datetime
import pandas as pd
from job import Job, Trace
from policy import (
    ShortestJobFirst,
    FirstInFirstOut,
    ShortestRemainingTimeFirst,
    QuasiShortestServiceFirst,
    Lucid,
    Tiresias,
)
from profiler import LeastGPUFirstProfiler
import random
import numpy as np

class GlobalScheduler:
    def __init__(self, global_policy, trace, cluster_num, clusters, placement,
                 log_dir, local_policy, logger, start_ts, estimator, updater):
        self.global_policy = global_policy  # 专家算法
        self.trace = trace
        self.cluster_num = cluster_num # 4
        self.clusters = clusters
        self.placement = placement
        self.log_dir = log_dir
        self.local_policy = local_policy # qssf Helios during x GPU卡数
        self.logger = logger
        self.start_ts = start_ts # 13219200
        self.estimator = estimator
        self.updater = updater

        self.scheduler_list = []
        self.total_job_num = self.trace.job_num()
        self.end_job_num = 0
        self.time = 0
        self._init_schedulers()

        self.run_list = []
        self.que_list = []

        # Time Sequence Recorder
        self.total_gpu_num =[]
        self.idle_gpu_num = []
        self.pend_gpu_num = []
        self.run_job_num = []
        self.pend_job_num = []
        self.pend_job_num_less_8 = []
        self.total_node_num = []
        self.consolidate_node_num = []
        self.shared_node_num = []  # >=2 jobs on one node
        self.sm_util = []
        self.gmem_util = []
        self.sm_util_active = []
        self.gmem_util_active = []

        self._rr_counter = 0

    def _init_schedulers(self):
        for i in range(self.cluster_num):
            cl = self.clusters[i]
            if self.local_policy == "sjf":
                scheduler = ShortestJobFirst(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            elif self.local_policy == "fifo":
                scheduler = FirstInFirstOut(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            elif self.local_policy == "srtf":
                scheduler = ShortestRemainingTimeFirst(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            elif self.local_policy == "qssf":
                scheduler = QuasiShortestServiceFirst(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            elif self.local_policy == "lucid":
                scheduler = Lucid(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            elif self.local_policy == "tiresias":
                scheduler = Tiresias(cl, self.placement, self.log_dir, self.logger, self.estimator, self.updater)
            else:
                raise ValueError(f"Unsupported scheduling policy: {self.local_policy}")
            
            self.scheduler_list.append(scheduler)

    def run(self):
        self.time = self.start_ts # 518400.0
        prev_index = 0
        job_list = self.trace.job_list # len = 7603

        while self.end_job_num != self.total_job_num:
            jobs_to_allocate = []

            for idx in range(prev_index, self.total_job_num):
                job = job_list[idx]
                if job["submit_time"] == self.time:  #job到达事件
                    status = self.seq_recorder()
                    cluster_idx = self.select_cluster(status)  # !!!
                    jobs_to_allocate.append((job, cluster_idx))
                    prev_index = idx + 1
                elif job["submit_time"] > self.time:
                    break
            self.end_job_num = 0
            for i in range(self.cluster_num): #4
                jobs_for_this_cluster = [job for job, cid in jobs_to_allocate if cid == i]
                if jobs_for_this_cluster:
                    self.end_job_num += self.scheduler_list[i].simulate(jobs_for_this_cluster, self.time)
                else:
                    self.end_job_num += self.scheduler_list[i].simulate(None, self.time)

            self.time += 1

            if self.time % 100 == 0:
                self.run_list = []
                self.que_list = []
                for i in range(self.cluster_num):
                    self.run_list += self.scheduler_list[i].run_list
                    self.que_list += self.scheduler_list[i].que_list
                self.logger.info(
                    f"Time: {int(self.time)} | "
                    f"Total Job: {self.total_job_num} | End job: {self.end_job_num}| Running job: {len(self.run_list)} | Pending job: {len(self.que_list)}"
                )
        self.logger.info(f"Finish")

        df = pd.DataFrame(self.trace.job_list)
        df["jct"] = df["end_time"] - df["submit_time"]
        avg_jct = round(df["jct"].mean(), 2)
        avg_que = round(df["queue"].mean(), 2)
        self.logger.info(f"Average JCT: {avg_jct} | Average Queue: {avg_que}")

        df.to_csv(f"{self.log_dir}/train_log.csv", index=False)

        return True
    
    def seq_recorder(self):
        self.total_gpu_num =[]
        self.idle_gpu_num = []
        self.pend_gpu_num = []
        self.run_job_num = []
        self.pend_job_num = []
        self.pend_job_num_less_8 = []
        self.total_node_num = []
        self.consolidate_node_num = []
        self.shared_node_num = []  # >=2 jobs on one node
        self.sm_util = []
        self.gmem_util = []
        self.sm_util_active = []
        self.gmem_util_active = []

        for i in range(self.cluster_num):
            self.total_gpu_num.append(self.scheduler_list[i].cl.total_gpus)                                     # 总GPU数
            self.idle_gpu_num.append(self.scheduler_list[i].cl.cluster_free_gpus())                             # 空闲GPU数量
            self.pend_gpu_num.append(sum(job.__getitem__("gpu_num") for job in self.scheduler_list[i].que_list)) # 队列中job需要GPU数量
            self.run_job_num.append(len( self.scheduler_list[i].run_list))                                       # 运行job数
            self.pend_job_num.append(len( self.scheduler_list[i].que_list))                                      # 排队job数
            self.pend_job_num_less_8.append(self.scheduler_list[i].pend_job_num_small())                         # 队列中需要GPU数目小于8的job数
            self.total_node_num.append( self.scheduler_list[i].cl.node_num)                                     # 总节点数
            self.consolidate_node_num.append( self.scheduler_list[i].cl.consolidate_node_num())                 # 只运行一个job的节点数
            self.shared_node_num.append( self.scheduler_list[i].cl.shared_node_num())                           # 运行job数大于1的节点数
        status = Status(self.total_gpu_num, self.idle_gpu_num, self.pend_gpu_num, self.run_job_num, self.pend_job_num, self.pend_job_num_less_8, self.total_node_num, self.consolidate_node_num, self.shared_node_num, self.sm_util, self.gmem_util, self.sm_util_active, self.gmem_util_active)
        return status

    def select_cluster(self, status):
        if self.global_policy == "round_robin":
            idx = self._rr_counter % self.cluster_num
            self._rr_counter += 1
            return idx
        
        if self.global_policy == "random":
            return random.randrange(self.cluster_num)
        
        if self.global_policy == "max_free_gpus":
            lst = status.idle_gpu_num
            return lst.index(max(lst))

        if self.global_policy == "min_load_first":
            lst = np.sum([np.array(status.total_gpu_num), -np.array(status.idle_gpu_num), np.array(status.pend_gpu_num)], axis=0) # 
            return int(np.argmin(lst))  # 返回最小值索引
        
class Status:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self, total_gpu_num, idle_gpu_num, pend_gpu_num, run_job_num, pend_job_num, pend_job_num_less_8, total_node_num, consolidate_node_num, shared_node_num, sm_util, gmem_util, sm_util_active, gmem_util_active):
        self.total_gpu_num = total_gpu_num
        self.idle_gpu_num = idle_gpu_num
        self.pend_gpu_num = pend_gpu_num
        self.run_job_num = run_job_num
        self.pend_job_num = pend_job_num
        self.pend_job_num_less_8 = pend_job_num_less_8
        self.total_node_num = total_node_num
        self.consolidate_node_num = consolidate_node_num
        self.shared_node_num = shared_node_num
        self.sm_util = sm_util
        self.gmem_util = gmem_util
        self.sm_util_active = sm_util_active
        self.gmem_util_active = gmem_util_active

def trace_profile(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts):
    profiler = LeastGPUFirstProfiler(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts)
    profiler.profile()
    trace.reset_trace()
    logger.info("Finish Profiling")
    return trace


def get_available_schedulers():
    return ["fifo", "sjf", "srtf", "qssf", "lucid", "tiresias"]


def get_sweep_schedulers():
    return ["fifo", "sjf", "srtf", "qssf", "tiresias"]


def get_available_placers():
    return ["random", "consolidate", "consolidateFirst"]


def trace_process(dir, date_range, read_full, cluster_name, cluster_num):
    start = "2020-04-01 00:00:00"
    if read_full == False:
        df_list = []
        for cn in cluster_name:
            df = pd.read_csv(
                dir + "/" + cn + "/cluster_log.csv",
                parse_dates=["submit_time"],
                usecols=["job_id", "user", "vc", "gpu_num", "cpu_num", "state", "submit_time", "duration"],
            )
            df_list.append(df)    
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv", # './data/Venus/cluster_full_log.csv'
            parse_dates=["submit_time"], # 自动将 submit_time 转为时间戳
            usecols=[
                "job_id",
                "user",
                "vc",
                "jobname",
                "gpu_num",
                "cpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]

    df.sort_values(by="submit_time", inplace=True)
    df = df[df["submit_time"] >= pd.Timestamp(start)] # 114749 筛选出大于2020-04-01提交的job
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x)))) 

    # Normalizing 以第一个job为时间 0 点
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part   所有提交时间在 2020-09-01 到 2020-09-26 的作业
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)] # 筛选出2020-09-01 到 2020-09-26的job

    accelaration_factor = 1
    df["submit_time"] = np.ceil(begin+(df["submit_time"]- begin)/accelaration_factor)

    df.reset_index(inplace=True, drop=True)

    return df, begin

def trace_real_process(dir):
    df = pd.read_csv(
        dir + "/cluster_full_log.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_pollux_process(dir, idx):
    df = pd.read_csv(
        f"{dir}/cluster_full_log_{idx}.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_philly_process(dir, date_range, read_full):
    start = "2017-10-01 00:00:00"
    if read_full == False:
        df = pd.read_csv(
            dir + "/cluster_log.csv",
            parse_dates=["submit_time"],
            usecols=["user", "vc", "jobname", "gpu_num", "state", "submit_time", "duration"],
        )
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv",
            parse_dates=["submit_time"],
            usecols=[
                "user",
                "vc",
                "job_id",
                "gpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]
    df.sort_values(by="submit_time", inplace=True)
    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    df.rename(columns={"jobname": "job_id"}, inplace=True)
    df["state"] = df["state"].replace("Pass", "COMPLETED")
    df["state"] = df["state"].replace("Failed", "FAILED")
    df["state"] = df["state"].replace("Killed", "CANCELLED")

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, begin


def trace_parser(df):
    trace = Trace()

    for _, series in df.iterrows():
        trace.append_job(Job(series))
    trace.sort_jobs("submit_time")
    return trace


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler()  # sys.stdout

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger


def cluster_concatenate(policy, placer, log_dir, dir):
    prefix = f"{policy}_{placer}"
    if not os.path.exists(log_dir + "/all"):
        os.mkdir(log_dir + "/all")

    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vcs = vc_df.index.to_list()

    """Log"""
    cluster_log = pd.DataFrame()
    for vc in vcs:
        vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
        cluster_log = pd.concat([cluster_log, vc_log])
    cluster_log.sort_values(by="submit_time", inplace=True)
    cluster_log.to_csv(f"{log_dir}/all/{prefix}_all_log.csv", index=False) # log/Venus_Sept/all/lucid_consolidate_all_log.csv

    """Seq"""
    cluster_seq = pd.DataFrame()
    add_list = [
        "total_gpu_num",
        "idle_gpu_num",
        "pending_gpu_num",
        "running_gpujob_num",
        "pending_gpujob_num",
        "pending_job_num_less_8",
        "total_node_num",
        "consolidate_node_num",
        "shared_node_num",
    ]
    for vc in vcs:
        vc_seq = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_seq.csv")
        if len(cluster_seq) == 0:
            cluster_seq = vc_seq
            continue
        cluster_seq[add_list] = cluster_seq[add_list] + vc_seq[add_list]
        cluster_seq.dropna(inplace=True)
        cluster_seq = cluster_seq.astype(int)
        cluster_seq["gpu_utilization"] = (
            (cluster_seq["total_gpu_num"] - cluster_seq["idle_gpu_num"]) / cluster_seq["total_gpu_num"]
        ).round(3)
    cluster_seq.to_csv(f"{log_dir}/all/{prefix}_all_seq.csv", index=False)   # log/Venus_Sept/all/lucid_consolidate_all_seq.csv


def cluster_analysis(placer, log_dir, dir):
    """Generate Algorithm Comparsion CSV"""
    # ignore_warm_up = start_ts + 7*24*3600

    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vcs = vc_df.index.to_list()
    vcs.append("all")

    files = os.listdir(f"{log_dir}/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))

    # prefix_list = []
    # for i in get_available_schedulers():
    #     prefix = f"{i}_{placer}"
    #     prefix_list.append(prefix)

    jct_avg = pd.DataFrame()
    que_avg = pd.DataFrame()
    for prefix in prefix_list:
        for vc in vcs:
            vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
            # vc_log = vc_log[vc_log['submit_time'] > ignore_warm_up]
            jct_avg.at[vc, prefix] = vc_log["jct"].mean()
            que_avg.at[vc, prefix] = vc_log["queue"].mean()

    jct_avg = jct_avg.astype(int)
    que_avg = que_avg.astype(int)
    jct_avg.to_csv(f"{log_dir}/jct_avg_{placer}.csv") # /root/Lucid/simulation/log/Venus_Sept/jct_avg_consolidate.csv
    que_avg.to_csv(f"{log_dir}/que_avg_{placer}.csv") # /root/Lucid/simulation/log/Venus_Sept/que_avg_consolidate.csv


def get_trace(experiment_name, trace_dir, read_full, cluster_name, cluster_num, idx=None):
    if "Philly" in experiment_name:
        trace_range = ("2017-10-01 00:00:00", "2017-10-07 23:59:00")
        trace_df, start_ts = trace_philly_process(trace_dir, trace_range, read_full)
    elif "Pollux" in experiment_name:
        trace_df, start_ts = trace_pollux_process(trace_dir, idx)
    else:
        if "train" in experiment_name:
            # trace_range = ("2020-04-07 00:00:00", "2020-07-31 23:59:00")
            trace_range = ("2020-04-07 00:00:00", "2020-04-07 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full, cluster_name, cluster_num) # './data'
        elif "eval" in experiment_name:
            trace_range = ("2020-08-01 00:00:00", "2020-09-26 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full, cluster_name, cluster_num)
        else:
            raise ValueError

    return trace_df, start_ts


def profiler_config(experiment_name, vc_dict): # experiment_name = 'Venus_Sept' vc_dict读取字vc_config
    cluster = experiment_name.split("_")[0]
    profile_scale = {"Venus": 2, "Philly": 2}
    profile_time = {"Venus": 200, "Philly": 80}
    profile_factor = {"Venus": 4, "Philly": 2}

    # Basic Config
    scale, time, factor = profile_scale[cluster], profile_time[cluster], profile_factor[cluster]
    if cluster == "Philly":
        vc_dict["philly"] -= scale
    elif cluster == "Venus":
        vc_dict["vc8Gr"] -= 1
        vc_dict["vcefl"] -= 1
        # vc_dict["vcYVn"] -= 1  # For elastic scaling
    return vc_dict, scale, time, factor  # 手动配置的参数


def check_profiler_scale_available(experiment_name, scale, vc_dict, prof_locate_vc=None):
    # Use only for debug
    default_vc = {
        "Venus": "vc8Gr",
        "Saturn": "vcqdr",
        "Philly": "philly",
    }
    cluster = experiment_name.split("_")[0]

    if not prof_locate_vc:
        vc = default_vc[cluster]

    if scale <= vc_dict[vc]:
        return vc
    else:
        raise ValueError("Profile Node Scale Exceed VC Capacity")


if __name__ == "__main__":
    files = os.listdir(f"log/Venus_Sept/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))
    print(prefix_list)
