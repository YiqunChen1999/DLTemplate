
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

from utils import utils
# import utils


def calc_F(foreground_mask, gt_mask, bound_th=0.008):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.
	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask         (ndarray): binary annotated image.
	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask)
	gt_boundary = seg2bmap(gt_mask)

	from skimage.morphology import binary_dilation,disk

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F_score = 0
	else:
		F_score = 2*precision*recall/(precision+recall)

	return F_score


def seg2bmap(seg,width=None, height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.
	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]
	Returns:
		bmap (ndarray):	Binary boundary map.
	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+floor((y-1)+height / h)
					i = 1+floor((x-1)+width  / h)
					bmap[j,i] = 1

	return bmap


def calc_jaccard(annotation, segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)


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

    def calc_metrics(self, phase, epoch, *args, **kwargs):
        F_score = calc_F(*args, **kwargs)
        jaccard = calc_jaccard(*args, **kwargs)
        self.record(phase, epoch, "F_score", F_score)
        self.record(phase, epoch, "Jaccard", jaccard)
        return F_score, jaccard

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
    pass
