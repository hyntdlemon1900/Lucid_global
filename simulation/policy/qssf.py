from .policy import Policy


class QuasiShortestServiceFirst(Policy):
    def __init__(self, vc, placement, log_dir, logger, estimator, updater):
        super(QuasiShortestServiceFirst, self).__init__(vc, placement, log_dir, logger)
        self.estimator = estimator
        self.updater = updater
        self._name = "qssf"

    def simulate(self, jobs, current_time):
        self.time = current_time
        if jobs is not None:
            for job in jobs:
                job["status"] = "pend"
                self.que_list.append(job)

        """1. Check & Release End Jobs"""
        run_ls = self.run_list.copy()  # Avoid list.remove() issue
        for job in run_ls:
            if self.time == job["end_time"]:
                job["remain"] = 0
                job["status"] = "end"
                self.end_job_num += 1
                assert self._vc.release_resource(job) == True
                self.run_list.remove(job)

        """3. Assign Priority If Exist Job Pending"""
        # NOTE: Sort by priority given by estimator -- QSSF
        # Only assign priority to the pending job, new job will sort by required gpu_num
        self.que_list.sort(key=lambda x: x.__getitem__("gpu_num")*x.__getitem__("duration"))
        que_ls = self.que_list.copy()  # Avoid list.remove() issue
        for job in que_ls:
            if self.job_placer(job):
                job["start_time"] = self.time
                job["end_time"] = job["start_time"] + job["duration"]
                job["queue"] = self.time - job["submit_time"]
                job["status"] = "run"
                self.que_list.remove(job)
                self.run_list.append(job)
            else:
                break

        return self.end_job_num
