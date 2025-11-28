from .policy import Policy


class ShortestJobFirst(Policy):
    def __init__(self, vc, placement, log_dir, logger, estimator, updater):
        super(ShortestJobFirst, self).__init__(vc, placement, log_dir, logger)
        self._name = "sjf"
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
            if self.time == job["end_time"]:
                job["remain"] = 0
                job["status"] = "end"
                self.end_job_num += 1
                assert self._vc.release_resource(job) == True
                self.run_list.remove(job)

        # Pend Job
        # NOTE: Sort by duration -- SJF
        self.que_list.sort(key=lambda x: x.__getitem__("duration"))
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
