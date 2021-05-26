
r"""
Author:
    Yiqun Chen
Docs:
    Help build loss functions.
"""

import os, sys, logging, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
# import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
import copy
import contextlib
from termcolor import colored
from configs.configs import cfg


class Logger:
    """
    Help user log infomation to file and | or console.
    """
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self._build_()

    def _build_(self):
        self.log_fns = {
            "info": logging.info, 
            "debug": logging.debug, 
            "warning": logging.warning, 
            "error": logging.error, 
            "critical": logging.critical, 
        }
        self.colors = {
            "info": None, 
            "debug": None, 
            "warning": "yellow", 
            "error": "red", 
            "critical": "red", 
        }
        t = time.gmtime()
        date = "{}-{}-{}-{}".format(
            str(t.tm_mon).zfill(2), 
            str(t.tm_mday).zfill(2), 
            str(t.tm_hour).zfill(2), 
            str(t.tm_min).zfill(2), 
        )
        self.path2logfile = os.path.join(cfg.log.dir, f"{date}.log")
        logging.basicConfig(
            level=logging.INFO, 
            format="[%(asctime)s] [%(filename)-20s: %(lineno)-4d] [%(levelname)-10s] %(message)s",
            handlers=[
                logging.FileHandler(filename=self.path2logfile, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.path2runs = os.path.join(f"{self.cfg.log.dir}/runs/{date}")
        self.writer = SummaryWriter(log_dir=self.path2runs)

    def log(self, level, msg):
        self.log_fns[level.lower()](msg)

    @contextlib.contextmanager
    def log_info(self, state: bool, level: str, msg: str):
        state_info = "[" + colored("{:<8}".format("RUNNING"), "green") + "]" if state else ""
        self.log(level, "{} {}".format(state_info, colored(msg, self.colors[level.lower()])))
        yield
        if state:
            state_info = "[" + colored("{:<8}".format("DONE"), "green") + "]" if state else ""
            self.log(level, "{} {}".format(state_info, colored(msg, self.colors[level.lower()])))

    def raise_error(self, exc, msg):
        self.log("error", msg)
        exc(msg)

    def close(self):
        self.writer.close()
        pass


logger = Logger(cfg)


if __name__ == "__main__":
    from configs.configs import cfg
    logger = Logger(cfg)
    with logger.log_info(True, "info", "hello world"):
        print("hello again.")
