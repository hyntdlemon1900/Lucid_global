import numpy as np


class Cluster:
    def __init__(self, cn, node_num, num_gpus_per_node, num_cpus_per_node):
        self.name = cn
        self._num_gpus_per_node = num_gpus_per_node
        self._num_cpus_per_node = num_cpus_per_node
        self.node_num = node_num  #！！！服务器 8张GPU

        self.base_node_num = node_num # 2
        self._num_gpus_per_node = num_gpus_per_node # 8 gpu
        self._num_cpus_per_node = num_cpus_per_node # 96 cpu core
        # Maintain both a list for iteration and a dict for O(1) node lookup
        self.node_list = []
        self._nodes = {}
        self.init_node() # 初始化2个node
        self.total_gpus = num_gpus_per_node * node_num
        self.total_cpus = num_cpus_per_node * node_num

        # Cached cluster-wide free resources (kept in sync with nodes)
        self._free_gpus = self.total_gpus
        self._free_cpus = self.total_cpus

        self.colocate_enable = 0
        # Temp Node num with additional temp_node_num_base
        self.temp_node_num_base = 9999
        self.has_temp_node = False  # To avoid first-time scaling error

    def cluster_free_gpus(self):
        """Return current free GPU count in O(1).

        Note: We maintain this as a cached value which is updated whenever
        node-level allocations or releases occur.
        """
        return self._free_gpus

    def cluster_free_cpus(self):
        """Return current free CPU-core count in O(1)."""
        return self._free_cpus

    def init_node(self):
        for i in range(self.node_num):
            node = Node(i, self._num_gpus_per_node, self._num_gpus_per_node, cluster=self)
            self.node_list.append(node)
            self._nodes[i] = node

    def check_node_inside(self, node_id):
        return node_id in self._nodes

    def check_node_inside_idle(self, node_id):
        node = self._nodes.get(node_id)
        if node is None:
            return False
        return node.free_gpus == self._num_gpus_per_node

    def add_new_node(self, change_node_num, force_same_node):
        for i in range(change_node_num):
            temp_node_num = i + self.temp_node_num_base
            if self.check_node_inside(temp_node_num) and force_same_node:
                # temp_node_num = temp_node_num + 1000
                # raise ValueError("Temp node num already exists")
                return False
            node = Node(temp_node_num, self._num_gpus_per_node, self._num_gpus_per_node, cluster=self)
            self.node_list.append(node)
            self._nodes[temp_node_num] = node
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num
        # Newly added nodes are fully free
        self._free_gpus += self._num_gpus_per_node * change_node_num
        self._free_cpus += self._num_cpus_per_node * change_node_num

    def exchange_node_status(self, idle_node, i):
        # Just for simple simulation implementation in some rare cases.
        # In reality, we can directly remove different nodes.
        assert idle_node.check_free_gpus() == self._num_gpus_per_node
        temp_id = self.temp_node_num_base + i
        temp_node = self.get_node(temp_id)
        old_idle_id = idle_node.node_name

        # Swap node ids in mapping first
        temp_node.update_node_name(old_idle_id)
        idle_node.update_node_name(temp_id)

        # Update mapping dict to reflect new ids
        self._nodes[old_idle_id] = temp_node
        self._nodes[temp_id] = idle_node

        temp_node.exchange_job_status()

    def remove_idle_node(self, change_node_num, force_same_node):
        idle_node_list = self.idle_node_list()
        if len(idle_node_list) < abs(change_node_num):
            return False  # Not enough idle nodes
        idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
        idle_node_list = idle_node_list[: abs(change_node_num)]
        for i in range(abs(change_node_num)):
            if idle_node_list[i].node_name < self.temp_node_num_base and force_same_node and self.has_temp_node:
                self.exchange_node_status(idle_node_list[i], i)
                idle_node_list = self.idle_node_list()
                idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
                assert idle_node_list[0].node_name >= self.temp_node_num_base
            to_remove_node = idle_node_list[i]
            self.node_list.remove(to_remove_node)
            # Remove from mapping
            self._nodes.pop(to_remove_node.node_name, None)
            # All resources on an idle node are free; removing reduces totals
            self._free_gpus -= self._num_gpus_per_node
            self._free_cpus -= self._num_cpus_per_node
        self.has_temp_node = True
        assert len(self.node_list) == self.node_num + change_node_num
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num
        return True

    def update_node(self, change_node_num, force_same_node=True):
        if change_node_num > 0:
            self.add_new_node(change_node_num, force_same_node)
        elif change_node_num < 0:
            self.remove_idle_node(change_node_num, force_same_node)
        else:
            raise ValueError("`change_node_num` should not be 0")

    def get_node(self, node_id):
        # Fast O(1) node lookup
        return self._nodes.get(node_id)

    def idle_node_list(self):
        idle_node_list = []
        for node in self.node_list:
            if node.free_gpus == self._num_gpus_per_node:
                idle_node_list.append(node)
        return idle_node_list

    def avail_node_list(self):
        avail_node_list = []
        for node in self.node_list:
            if node.free_gpus > 0:
                avail_node_list.append(node)
        return avail_node_list

    def release_resource(self, job):
        nodes_list = job["nodes"]
        for dict in nodes_list:
            for i, gpu_list in dict.items():
                node = self.get_node(i)
                assert node.node_name == i
                assert node.release_gpu(gpu_list, job)
        return True

    def check_colocate_jobs(self, job):
        # nodes_list = job["nodes"]
        # recover_jobs = set()
        # for dict in nodes_list:
        #     for i, gpu_list in dict.items():
        #         node = self.node_list[i]
        #         jobs = node.check_colocate_jobs(gpu_list, job)
        #         recover_jobs |= set(jobs)
        # return list(recover_jobs)
        nodes_list = job["nodes"]
        dict = nodes_list[0]
        for i, gpu_list in dict.items():
            node = self.get_node(i)
            colo_job_id = node.check_colocate_jobs(gpu_list, job)
            if colo_job_id:
                return colo_job_id
            else:
                raise NotImplementedError

    # Only one job running in a node
    def consolidate_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num == 1:
                list.append(node)
        return len(list)

    def shared_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num > 1:
                list.append(node)
        return len(list)

    def check_sm_util(self):
        list = []
        for node in self.node_list:
            list.append(node.check_avg_gpu_util())
        return np.mean(list)

    def check_gmem_util(self):
        list = []
        for node in self.node_list:
            list.append(node.check_avg_mem_util())
        return np.mean(list)

    def check_active_sm_util(self):
        list = []
        for node in self.node_list:
            util = node.check_active_avg_gpu_util()
            if util:
                list.append(util)
        if list:
            return np.mean(list)
        else:
            return 0

    def check_active_gmem_util(self):
        list = []
        for node in self.node_list:
            util = node.check_active_avg_mem_util()
            if util:
                list.append(util)
        if list:
            return np.mean(list)
        else:
            return 0

class Node:
    def __init__(self, node_name, num_gpus, num_cpus, cluster=None):
        # Optional back-reference to owning Cluster to update cached totals
        self.cluster = cluster
        self.node_name = node_name
        self.num_gpus = num_gpus # 8
        self.num_cpus = num_cpus # 8
        self.previous_node_name = node_name

        self.job_num = 0
        self.free_cpus = num_cpus
        # dynamic cache of free GPUs, kept in sync with node_gpu_dict
        self.free_gpus = num_gpus
        # self.colocate_gpu_num = 0

        # Mapping: job_id -> list of gpu indices on this node
        self.node_job_dict = {}
        # Mapping: gpu index -> list of job dicts using this GPU
        self.node_gpu_dict = self.init_gpu_dict()
        # GPU Utilization and Memory Usage per GPU
        self.node_gutil = self.init_gpu_state()  # GPU Utilization
        self.node_gmem = self.init_gpu_state()  # GPU Memory Usage

    def init_gpu_dict(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: []})
        return gdict

    def init_gpu_state(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: 0})
        return gdict

    def check_avg_gpu_util(self):
        return np.mean(list(self.node_gutil.values()))

    def check_avg_mem_util(self):
        return np.mean(list(self.node_gmem.values()))

    def check_active_avg_gpu_util(self):
        gutils = list(self.node_gutil.values())
        active_ls = list(filter(lambda x: x > 0, gutils))
        if active_ls:
            return np.mean(active_ls)
        else:
            return False

    def check_active_avg_mem_util(self):
        gmems = list(self.node_gmem.values())
        active_ls = list(filter(lambda x: x > 0, gmems))
        if active_ls:
            return np.mean(active_ls)
        else:
            return False

    def check_free_gpus(self):
        # Rely on cached value instead of scanning dict
        return self.free_gpus

    def check_free_gpu_list(self):
        return [k for k, v in self.node_gpu_dict.items() if not v]

    def check_colocate_gpu_list(self):
        return [k for k, v in self.node_gpu_dict.items() if len(v) == 2]

    def check_colocate_jobs(self, gpu_list, job):
        # colocate_jobs = set()
        # for i in gpu_list:
        #     for j in self.node_gpu_dict[i]:
        #         if j is not job:
        #             colocate_jobs.add(j)
        # return list(colocate_jobs)
        # colocate_jobs = set()
        target = None
        # Find any job (other than the given one) that occupies exactly gpu_list
        for k, v in self.node_job_dict.items():
            if v == gpu_list and k != job["job_id"]:
                target = k
                break
        if target is not None:
            return target
        return None

    """colocate usage"""

    def allocate_colocate_gpu(self, gpu_list, job, gutil, gmem):
        # num_gpu = len(gpu_list)
        # self.colocate_gpu_num += num_gpu
        self.job_num += 1

        for i in gpu_list:
            jobs_on_gpu = self.node_gpu_dict[i]
            assert len(jobs_on_gpu) == 1
            jobs_on_gpu.append(job)
            self.node_gutil[i], self.node_gmem[i] = gutil, gmem
        self.node_job_dict[job["job_id"]] = list(gpu_list)
        return True

    """allocate"""

    def allocate_gpu(self, num_gpu, job):
        assert num_gpu <= self.free_gpus
        self.free_gpus -= num_gpu
        # Update cluster cached free GPU count if available
        if self.cluster is not None:
            self.cluster._free_gpus -= num_gpu
        self.job_num += 1
        allocate_gpus = []
        toallocate = num_gpu
        for k, v in self.node_gpu_dict.items():
            if toallocate == 0:
                break
            if not v:
                allocate_gpus.append(k)
                v.append(job)
                # self.node_gutil[k] = job["gpu_util"]
                # self.node_gmem[k] = job["gmem"]
                toallocate -= 1
        assert num_gpu == len(allocate_gpus)
        self.node_job_dict[job["job_id"]] = allocate_gpus
        return allocate_gpus

    """release"""

    def release_gpu(self, gpu_list, job):
        if job["exclusive"] > 0:
            assert self.free_gpus + len(gpu_list) <= self.num_gpus
            self.free_gpus += len(gpu_list)
            if self.cluster is not None:
                self.cluster._free_gpus += len(gpu_list)
            self.job_num -= 1
        else:
            # assert self.colocate_gpu_num >= num_gpu
            # self.colocate_gpu_num -= len(gpu_list)
            self.job_num -= 1

        for i in gpu_list:
            assert isinstance(i, int)
            self.node_gpu_dict[i].remove(job)
            if self.node_gpu_dict[i] == []:
                self.node_gutil[i] = 0
                self.node_gmem[i] = 0
            else:
                assert len(self.node_gpu_dict[i]) == 1
                exist_job = self.node_gpu_dict[i][0]
                self.node_gutil[i] = exist_job["gpu_util"]
                self.node_gmem[i] = exist_job["gmem"]

        self.node_job_dict.pop(job["job_id"])

        return True

    def update_node_name(self, new_name):
        # Echo `exchange_node_status`
        self.previous_node_name = self.node_name
        self.node_name = new_name

    def exchange_job_status(self,):
        # Echo `exchange_node_status`
        jobs = []
        for k, v in self.node_gpu_dict.items():
            if v != []:
                for job in v:
                    if job not in jobs:
                        jobs.append(job)
        for job in jobs:
            for allocate_dict in job["nodes"]:
                k, v = list(allocate_dict.items())[0]
                if k == self.previous_node_name:
                    new_dict = {self.node_name: v}
                    job["nodes"].remove(allocate_dict)
                    job["nodes"].append(new_dict)

    # Future Extension
    def allocate_cpu(self, num_cpu):
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            if self.cluster is not None:
                self.cluster._free_cpus -= num_cpu
            return True

    def release_cpu(self, num_cpu):
        assert self.free_cpus + num_cpu <= self.num_cpus
        self.free_cpus += num_cpu
        if self.cluster is not None:
            self.cluster._free_cpus += num_cpu
        return True
