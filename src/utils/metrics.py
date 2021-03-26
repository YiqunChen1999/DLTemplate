
r"""
Author:
    Yiqun Chen
Docs:
    Metrics.
Note: 
    Adapted from https://github.com/fperazzi/davis-2017
"""

import os, sys, copy, torch
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from . import utils
from alphaconfig import AlphaConfig

_METRICS_ = {}

def add_metric_handler(metric_handler):
	_METRICS_[metric_handler.__name__.replace("calc_", "")] = metric_handler



@add_metric_handler
def calc_ssim(pred, anno):
	utils.raise_error(NotImplementedError, "ssim is not implemented.")


@add_metric_handler
def calc_psnr(pred, anno):
	utils.raise_error(NotImplementedError, "psnr is not implemented")


class MetricsHandler:
    def __init__(self, metrics=None):
		super(MetricsHandler, self).__init__()
		self.metrics = AlphaConfig()
		self._items_ = []
		for m in metrics:
			if m not in _METRICS_.keys():
				utils.raise_error(NotImplementedError, "Handler of metric {} is not in implemented metrics set {}".format(m, _METRICS_))
			self._items_.append(m)

	def register(self, metric):
		if m not in _METRICS_.keys():
			utils.raise_error(NotImplementedError, "Handler of metric {} is not in implemented metrics set {}".format(m, _METRICS_))
		self._items_.append(m)

    def update(self, phase, epoch, metric, value):
        if phase not in self.metrics.keys():
            self.metrics[phase] = {}
        if epoch not in self.metrics[phase].keys():
            self.metrics[phase][epoch] = {}
        if metric not in self.metrics[phase][epoch].keys():
			self.metrics[phase][epoch][metric] = AlphaConfig(value=0, counts=0)
		self.metrics[phase][epoch][metric].counts += 1
		counts = self.metrics[phase][epoch][metric].counts
		pre_value = self.metrics[phase][epoch][metric].value
		new_value = value / counts + pre_value * (counts - 1) / counts
		self.metrics[phase][epoch][metric].value = new_value
		return new_value

    def calc_metrics(self, phase, epoch, *args, **kwargs):
        for metric in self._items_:
			value = _METRICS_[metric](*args, **kwargs)
			self.update(phase, epoch, metric, value)
        return self.metrics[phase][epoch]


if __name__ == "__main__":
    pass
