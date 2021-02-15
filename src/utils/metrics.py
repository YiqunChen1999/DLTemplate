
r"""
Author:
    Yiqun Chen
Docs:
    Metrics.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import copy, skimage, math, sklearn
import numpy as np
from sklearn import metrics
from skimage import metrics
import matplotlib.pyplot as plt
import lpips, torch
import torch.nn.functional as F

from configs.configs import cfg
from utils import utils
# import utils

def calc_mae(y_true, y_pred, *args, **kwargs):
    if y_true.shape[0] == 3:
        y_true = np.transpose(y_true, (1, 2, 0))
        y_pred = np.transpose(y_pred, (1, 2, 0))
    mae_0 = sklearn.metrics.mean_absolute_error(y_true[:,:,0], y_pred[:,:,0])
    mae_1 = sklearn.metrics.mean_absolute_error(y_true[:,:,1], y_pred[:,:,1])
    mae_2 = sklearn.metrics.mean_absolute_error(y_true[:,:,2], y_pred[:,:,2])
    return np.mean([mae_0,mae_1,mae_2])

def calc_psnr(image_true, image_test, data_range=None, *args, **kwargs):
    psnr = skimage.metrics.peak_signal_noise_ratio(image_true, image_test, data_range=data_range)
    return psnr
    
def calc_ssim(im1, im2, data_range=None, multichannel=True, *args, **kwargs):
    if isinstance(im1, torch.Tensor):
        return utils.cal_ssim_pt(im1, im2, data_range, multichannel, *args, **kwargs)
    if im1.shape[0] == 3:
        im1 = np.transpose(im1, (1, 2, 0))
        im2 = np.transpose(im2, (1, 2, 0))
    ssim = skimage.metrics.structural_similarity(im1, im2, data_range=data_range, multichannel=multichannel)
    return ssim

lpips_alex_scorer = lpips.LPIPS()
lpips_alex_scorer, device = utils.set_device(lpips_alex_scorer, cfg.GENERAL.GPU)

def calc_lpips(image_true, image_test):
    r"""
    Info:
        Calculate LPIPS loss (AlexNet).
    Args:
        - image_true (ndarray | Tensor): takes values from [0, 1], should be RGB image with format [C, H, W].
        - image_test (ndarray | Tensor): takes values from [0, 1], should be RGB image with format [C, H, W].
    Returns:
        - score (float): the LPIPS loss.
    """
    
    if isinstance(image_true, np.ndarray):
        image_true = torch.from_numpy((image_true-0.5)*2)
        image_test = torch.from_numpy((image_test-0.5)*2)
    image_true = image_true.unsqueeze(0)
    image_test = image_test.unsqueeze(0)
    assert image_test.shape[1] == 3 and image_true.shape[1] == 3, \
        "Image should be RGB and format should be [C, H, W]"
    score = lpips_alex_scorer(image_true, image_test)
    score = score.item()
    return score

class Metrics:
    def __init__(self):
        self.metrics = {}

    def record(self, phase, epoch, item, value):
        if phase not in self.metrics.keys():
            self.metrics[phase] = {}
        if epoch not in self.metrics[phase].keys():
            self.metrics[phase][epoch] = {}
        if item not in self.metrics[phase][epoch].keys():
            self.metrics[phase][epoch][item] = []
        self.metrics[phase][epoch][item].append(value)

    def get_metrics(self, phase=None, epoch=None, item=None):
        metrics = copy.deepcopy(self.metrics)
        if phase is not None:
            metrics = {phase: metrics[phase]}
        if epoch is not None:
            for _p in metrics.keys():
                metrics[_p] = {epoch: metrics[_p][epoch]}
        if item is not None:
            for _p in metrics.keys():
                for _e in metrics[_p].keys():
                    metrics[_p][_e] = {item: metrics[_p][_e][item]}
        return metrics

    def mean(self, phase, epoch, item=None):
        mean_metrics = {}
        metrics = self.get_metrics(phase=phase, epoch=epoch, item=item)
        metrics = metrics[phase][epoch]
        for key, value in metrics.items():
            mean_metrics[key] = np.mean(np.array(value))
        return mean_metrics

    def cal_metrics(self, phase, epoch, *args, **kwargs):
        mae = cal_mae(*args, **kwargs)
        ssim = cal_ssim(*args, **kwargs)
        psnr = cal_psnr(*args, **kwargs)
        self.record(phase, epoch, "MAE", mae)
        self.record(phase, epoch, "SSIM", ssim)
        self.record(phase, epoch, "PSNR", psnr)
        return mae, ssim, psnr

    def plot(self, path2dir):
        for phase in self.metrics.keys():
            for item in self.metrics[phase][0].keys():
                values = []
                for epoch in self.metrics[phase].keys():
                    values.extend(self.metrics[phase][epoch][item])
                plt.plot(np.cumsum(np.ones(len(values))), np.array(values))
                plt.xlabel("epoch")
                plt.ylabel(item)
                plt.title("{} until epoch {}".format(item, epoch))
                plt.grid(b=True, which="both")
                save_dir = os.path.join(path2dir, phase, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(os.path.join(save_dir, "{}_{}_epoch_{}.png".format(phase, item, str(epoch).zfill(4))))
                plt.clf()
        


if __name__ == "__main__":
    # metrics_logger = Metrics()
    # for phase in ["train", "valid", "test"]:
    #     for epoch in range(20):
    #         for item in ["mse", "psnr", "ssim"]:
    #             for i in range(10):
    #                 metrics_logger.record(phase, epoch, item, np.random.randn(1))
    # metrics_logger.get_metrics()
    # print(metrics_logger.mean("train", 0, item=None))
    # metrics_logger.plot("/home/chenyiqun/models/dehazing/tmp")
    import cv2, time
    path2img1 = "/home/chenyiqun/tmp/HAZE2021/src/02.png"
    path2img2 = "/home/chenyiqun/tmp/HAZE2021/trg/02.png"
    img1 = (cv2.imread(path2img1, -1) / (2 ** 8-1)).astype(np.float32)
    img2 = (cv2.imread(path2img2, -1) / (2 ** 8-1)).astype(np.float32)

    print(cal_ssim(img1, img2, data_range=1.0))
    print(cal_ssim(torch.from_numpy(img1), torch.from_numpy(img2), data_range=1.0))

    # start = time.time()
    # for i in range(100):
    #     img1 = torch.rand((3, 256, 256))
    #     img2 = torch.rand((3, 256, 256))
    #     img1[img1 < -1.] = -1.
    #     img2[img2 < -1.] = -1.
    #     img1[img1 > 1.] = 1.
    #     img2[img2 > 1.] = 1.
    #     # print(cal_lpips(img1.cuda(), img2.cuda()))
    # print(time.time() - start)
    
    # print(cal_mae(img1, img2))
    # print(cal_ssim(img1, img2, data_range=1, multichannel=True))
    # print(cal_psnr(img1, img2, data_range=1))

    # 0.031292318003256575 25.087839612030397 0.8068734330381345
    # 0.031292318003256575 25.087839612030397 0.8068734330381345
