
r"""
Author:
    Yiqun Chen
Docs:
    Help build loss functions.
"""

import os, sys, logging, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
# from torch.utils.tensorboard import SummaryWriter
import copy


class Logger:
    """
    Help user log infomation to file and | or console.
    """
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.path2logfile = os.path.join(cfg.LOG.DIR, "{}.log".format(cfg.GENERAL.ID))
        self.enable_file_log = True
        if self.enable_file_log:
            logging.basicConfig(filename=self.path2logfile, level=logging.INFO, format="%(message)s")
        self._build_()

    def _build_(self):
        t = time.gmtime()
        self.path2runs = os.path.join("{}/{}/Mon{}Day{}Hour{}Min{}".format(
            self.cfg.LOG.DIR, 
            "runs", 
            str(t.tm_mon).zfill(2), 
            str(t.tm_mday).zfill(2), 
            str(t.tm_hour).zfill(2), 
            str(t.tm_min).zfill(2), 
        ))
        self.metrics = dict()
        # self.writer = SummaryWriter(log_dir=self.path2runs)

    def log_info(self, msg):
        if self.enable_file_log:
            logging.info(msg)
        print(msg)

    def close(self):
        # self.writer.close()
        pass


if __name__ == "__main__":
    pass
