
r"""
Author:
    Yiqun Chen
Docs:
    Metrics.
"""

import os, sys, copy, torch
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from . import utils
from alphaconfig import AlphaConfig

METRICS_ = {}

def add_metric_handler(metric_handler):
    METRICS_[metric_handler.__name__.replace("calc_", "")] = metric_handler
    return metric_handler


@add_metric_handler
def calc_ssim(img1, img2, data_range, multichannel=True, *args, **kwargs):
    ssim = utils.calc_ssim(img1, img2, data_range=data_range, multichannel=multichannel, *args, **kwargs)
    return ssim


@add_metric_handler
def calc_psnr(img1, img2, data_range, *args, **kwargs):
    r"""
    Info:
        img1 (Tensor): 
        img2 (Tensor): 
        data_range (float): the possible max value of input image.
    """
    psnr = utils.calc_psnr(img1, img2, data_range, *args, **kwargs)
    return psnr


class MetricsHandler:
    def __init__(self, metrics=None):
        super(MetricsHandler, self).__init__()
        self.metrics = AlphaConfig()
        self._items_ = []
        for m in metrics:
            if m not in METRICS.keys():
                utils.raise_error(NotImplementedError, "Handler of metric {} is not in implemented metrics set {}".format(m, _METRICS_))
            self._items_.append(m)

    def register(self, metric):
        if m not in METRICS.keys():
            utils.raise_error(NotImplementedError, "Handler of metric {} is not in implemented metrics set {}".format(m, _METRICS_))
        self._items_.append(m)

    def update(self, dataset, phase, epoch, metric, value):
        epoch = str(epoch)
        if dataset not in self.metrics.keys():
            self.metrics[dataset] = {}
        if phase not in self.metrics[dataset].keys():
            self.metrics[dataset][phase] = {}
        if epoch not in self.metrics[dataset][phase].keys():
            self.metrics[dataset][phase][epoch] = {}
        if metric not in self.metrics[dataset][phase][epoch].keys():
            self.metrics[dataset][phase][epoch][metric] = AlphaConfig(value=0, counts=0)
        self.metrics[dataset][phase][epoch][metric].counts += 1
        counts = self.metrics[dataset][phase][epoch][metric].counts
        pre_value = self.metrics[dataset][phase][epoch][metric].value
        new_value = value / counts + pre_value * (counts - 1) / counts
        self.metrics[dataset][phase][epoch][metric].value = new_value
        return new_value

    def calc_metrics(self, dataset, phase, epoch, out, trg, data_range, *args, **kwargs):
        kwargs["device"] = out.device
        for metric in self._items_:
            value = METRICS[metric](out, trg, data_range=data_range, *args, **kwargs)
            self.update(dataset, phase, epoch, metric, value)
        return self.metrics[dataset][phase][epoch]

    def summarize(self, dataset, phase, epoch, logger=None, *args, **kwargs):
        fmt = "{:<20}" * (len(self._items_) + 1 )
        values = [dataset]
        for idx in range(len(self._items_)):
            values.append(str(round(self.metrics[dataset][phase][epoch][self._items_[idx]].value.item(), 5)))
        info = fmt.format("dataset", *self._items_)
        data = fmt.format(*values)
        msg = "\n========Metrics========\n" + info + "\n" + data
        utils.notify(msg, logger=logger)




if __name__ == "__main__":
    pass
