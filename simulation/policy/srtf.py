from .policy import Policy


class ShortestRemainingTimeFirst(Policy):
    def __init__(self, vc, placement, log_dir, logger, estimator, updater):
        super(ShortestRemainingTimeFirst, self).__init__(vc, placement, log_dir, logger)
        self._name = "srtf"
        self.estimator = estimator
        self.updater = updater

    def simulate(self, jobs, current_time):
        self.time = current_time
        if jobs is not None:
            for job in jobs:
                job["status"] = "pend"
                self.que_list.append(job)


        """1. Check & Release End Jobs"""
        run_ls = self.run_list.copy()  # Avoid list.remove() issue
        for job in run_ls:
            if job["remain"] == 0:
                job["status"] = "end"
                job["end_time"] = self.time
                self.end_job_num += 1
                assert self._vc.release_resource(job) == True
                self.run_list.remove(job)
            else:
                job["remain"] -= 1

        """3. Select Job to Preempt or Run """
        # NOTE: Sort by remain -- SRTF

        current_job = self.que_list + self.run_list
        current_job.sort(key=lambda x: x.__getitem__("remain"))

        quota = self._vc.total_gpus
        preempt_list = []
        prerun_list = []
        for job in current_job:
            if job.__getitem__("gpu_num") <= quota:
                quota -= job.__getitem__("gpu_num")
                if job["status"] == "pend":
                    prerun_list.append(job)
            elif job["status"] == "run":
                preempt_list.append(job)

        """4. Preempt Job """
        for job in preempt_list:
            job["ckpt_times"] += 1
            job.set_ckpt_time(self.time)
            job["status"] = "pend"
            job["remain"] += self.ckpt_overhead(job)
            assert self._vc.release_resource(job) == True
            job["nodes"] = []

            if job not in self.que_list:
                self.que_list.append(job)
            if job in self.run_list:
                self.run_list.remove(job)

        """5. Allocate Job """
        for job in prerun_list:
            if self.job_placer(job):
                job["status"] = "run"
                if job["ckpt_times"] == 0:
                    job["start_time"] = self.time
                    job["queue"] = self.time - job["submit_time"]
                else:
                    job["queue"] = job["queue"] + (self.time - job.get_ckpt_time())

                if job in self.que_list:
                    self.que_list.remove(job)
                if job not in self.run_list:
                    self.run_list.append(job)
            else:
                # May place fail because consolidate requirement
                if job not in self.que_list:
                    self.que_list.append(job)
                continue

        return self.end_job_num
