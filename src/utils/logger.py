
"""
Author  Yiqun Chen
Docs    Logger to record information, should not call other custom modules.
"""

import os, sys, logging, time
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))

class Logger:
    """
    Help user log infomation to file and | or console.
    """
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.path2logfile = os.path.join(cfg.LOG.DIR, "{}.log".format(cfg.GENERAL.ID))
        logging.basicConfig(filename=self.path2logfile, encoding='utf-8', level=logging.DEBUG, format='[%(asctime)s] %(message)s')

    def log_info(self, msg):
        logging.info(msg)
        msg = "[{:<20}] [{:<8}] {}".format(time.asctime(), "INFO", msg)
        print(msg)
        raise NotImplementedError("Method Logger.log_info does not implemented yet.")

