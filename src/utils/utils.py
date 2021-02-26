
r"""
Author:
    Yiqun Chen
Docs:
    Utilities, should not call other custom modules.
"""

import os, sys, copy, functools, time, contextlib, math
import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    log = print if logger is None else logger.log_info
    _state = "[{:<8}]".format("RUNNING") if state else ""
    log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))
    yield
    if state:
        _state = "[{:<8}]".format("DONE") if state else ""
        log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))


def log_info_wrapper(msg, logger=None):
    r"""
    Decorate factory.
    """
    def func_wraper(func):
        r"""
        The true decorate.
        """
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # log = print if logger is None else logger.log_info
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "RUNNING"), msg)
            with log_info(msg=msg, level="INFO", state=True, logger=logger):
                res = func(*args, **kwargs)
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "DONE"), msg)
            return res
        return wrapped_func
    return func_wraper


def resize_and_pad(img, resol, is_mask):
    r"""
    Info:
        Resize and pad image with zeros.
    Args:
        img (Ndarray | Tensor):
        resol (list | tuple): [H, W]
        is_mask (bool):
    Rets:
        img (Ndarray | Tensor): the type depends on input img's type.
    """
    dim = len(img.shape)
    org_resol = img.shape[-2: ]
    scale_factor = min(resol[0]/org_resol[0], resol[1]/org_resol[1])
    if isinstance(img, np.ndarray):
        org_type = "Ndarray"
        img = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        org_type = "Tensor"
    else:
        raise TypeError("Unexpected img type {}".format(type(img)))
    img = img.type(torch.float)
    if dim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif dim == 3:
        img = img.unsqueeze(0)
    img = F.interpolate(img, scale_factor=scale_factor, recompute_scale_factor=False)
    if is_mask:
        max_pix = torch.max(img)
        img = (img > (max_pix / 2)) * max_pix
    img_resol = img.shape[-2: ]
    # TODO Padding.
    padding_left = (resol[0] - img_resol[0]) // 2
    padding_right = resol[0] - padding_left
    padding_top = (resol[1] - img_resol[1]) // 2
    padding_bottom = resol[1] - padding_top
    img = F.pad(img, pad=(padding_left, padding_right, padding_top, padding_bottom))
    if org_type == "Ndarray":
        img = img.numpy()
    return img


def get_video_spatial_feature(featmap_H, featmap_W):
    spatial_batch_val = np.zeros((8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w] = [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
    return spatial_batch_val


def get_spatial_feats():
    spatial_map_s = get_video_spatial_feature(cfg.DATA.VIDEO.RESOLUTION[0][0], cfg.DATA.VIDEO.RESOLUTION[0][1])
    spatial_map_s = torch.from_numpy(spatial_map_s).unsqueeze(0)
    spatial_map_s = spatial_map_s.to(device=device)
    spatial_map_m = get_video_spatial_feature(cfg.DATA.VIDEO.RESOLUTION[1][0], cfg.DATA.VIDEO.RESOLUTION[1][1])
    spatial_map_m = torch.from_numpy(spatial_map_m).unsqueeze(0)
    spatial_map_m = spatial_map_m.to(device=device)
    spatial_map_l = get_video_spatial_feature(cfg.DATA.VIDEO.RESOLUTION[2][0], cfg.DATA.VIDEO.RESOLUTION[2][1])
    spatial_map_l = torch.from_numpy(spatial_map_l).unsqueeze(0)
    spatial_map_l = spatial_map_l.to(device=device)
    spatial_feats = [spatial_map_s, spatial_map_m, spatial_map_l]
    return spatial_feats


def get_coord(mask, normalize=True):
    shape = mask.shape
    assert len(shape) == 2, "Two many indices"
    if torch.sum(mask) == 0:
        return torch.tensor([0, 0, 1, 1]).unsqueeze(0).type(torch.float32)
    coord = torch.nonzero(mask)
    x_min = torch.min(coord[:, 1]).type(torch.float32)
    x_max = torch.max(coord[:, 1]).type(torch.float32)
    y_min = torch.min(coord[:, 0]).type(torch.float32)
    y_max = torch.max(coord[:, 0]).type(torch.float32)
    x_min /= shape[1]
    x_max /= shape[1]
    y_min /= shape[0]
    y_max /= shape[0]
    return torch.tensor([x_min, x_max, y_min, y_max]).unsqueeze(0)


def inference(model, data, device, *args, **kwargs):
    r"""
    Info:
        Inference once, without calculate any loss.
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
    """
    def _inference_V1(model, data, device, *args, **kwargs):
        frames = data[0]
        batch_size = frames.shape[0]
        gt_mask_s, gt_mask_m, gt_mask_l, bert = data[1]["mask_s"], data[1]["mask_m"], data[1]["mask_l"], data[1]["bert"]
        
        frames, gt_mask_s, gt_mask_m, gt_mask_l, bert = \
            frames.to(device), gt_mask_s.to(device), gt_mask_m.to(device), gt_mask_l.to(device), bert.to(device)

        # mask_s, mask_m, mask_l = model(txt, frames, **kwargs)
        mask_s, mask_m, mask_l, bbox_s, bbox_m, bbox_l = model(bert, frames, **kwargs)

        gt_bbox_s = torch.cat([get_coord(gt_mask_s[i]) for i in range(batch_size)], dim=0).to(device)
        gt_bbox_m = torch.cat([get_coord(gt_mask_m[i]) for i in range(batch_size)], dim=0).to(device)
        gt_bbox_l = torch.cat([get_coord(gt_mask_l[i]) for i in range(batch_size)], dim=0).to(device)
        
        outputs = {
            "mask_s": mask_s, "mask_m": mask_m, "mask_l": mask_l, 
            "bbox_s": bbox_s, "bbox_m": bbox_m, "bbox_l": bbox_l, 
        }
        targets = {
            "gt_mask_s": gt_mask_s, "gt_mask_m": gt_mask_m, "gt_mask_l": gt_mask_l, 
            "gt_bbox_s": gt_bbox_s, "gt_bbox_m": gt_bbox_m, "gt_bbox_l": gt_bbox_l, 
        }
        return outputs, targets

    return _inference_V1(model, data, device, *args, **kwargs) 


def inference_and_calc_loss(model, data, loss_fn, device, *args, **kwargs):
    r"""
    Info:
        Execute inference and calculate loss, sychronize the train and evaluate progress. 
    Args:
        - model (nn.Module):
        - data (dict): 
        - loss_fn (callable): function or callable instance.
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
        - loss (Tensor): calculated loss.
    """
    def _infer_and_calc_loss_V1(model, data, loss_fn, device, *args, **kwargs):
        outputs, targets = inference(model, data, device, *args, **kwargs)
        loss = loss_fn(outputs, targets)
        return outputs, targets, loss

    return _infer_and_calc_loss_V1(model, data, loss_fn, device, *args, **kwargs)


def calc_and_record_metrics(phase, epoch, outputs, targets, metrics_logger, logger=None):
    # outputs = [out.detach().cpu().numpy() for out in outputs]
    # targets = [trg.detach().cpu().numpy() for trg in targets]
    out = outputs["mask_l"].detach().cpu().numpy()
    trg = targets["gt_mask_l"].detach().cpu().numpy()
    batch_size = output[0].shape[0]
    for idx in range(batch_size):
        metrics_logger.cal_metrics(phase, epoch, trg[idx], out[idx])


def rgb2hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Info:
        Convert an image from RGB to HSV. The image data is assumed to be in the range of (0, 1).
    Args:
        - image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        - eps (float, optional): scalar to enforce numarical stability. Default: 1e-6.
    Returns:
        - (torch.Tensor): HSV version of the image with shape of :math:`(*, 3, H, W)`.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb2hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + eps)

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / (deltac + eps)

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def flip_and_rotate(img):
    r"""
    Info:
        Flip and rotate input image.
    Args:
        - img (Tensor)
    Returns:
        - imgs (list of Tensor)
    """
    imgs = []
    imgs.extend(rotate(img))
    imgs.extend(rotate(imgs[1]))
    return imgs


def rotate(img):
    imgs = []
    imgs.append(torch.rot90(torch.clone(img), k=1, dims=[1, 2]))
    imgs.append(torch.rot90(torch.clone(img), k=2, dims=[1, 2]))
    imgs.append(torch.rot90(torch.clone(img), k=3, dims=[1, 2]))
    return imgs


def crop(img, trg_h, trg_w, stride):
    r"""
    Info:
        Crop patches from a given image.
    Args:
        - img (ndarray): RGB (with shape H, W, 3).
        - trg_h (int): height of target patches.
        - trg_w (int): width of target patches.
        - stride (int): 
    Returns:
        - patches (list of ndarray):
    """
    assert 2 <= len(img.shape) <= 3, "Incorrect image shape."
    if len(img.shape) == 3:
        assert img.shape[-1] == 3, "Incorrect format of RGB image."
    src_h, src_w = img.shape[0], img.shape[1]
    
    img = torch.from_numpy(img)
    pad_t = int((np.ceil(src_h / trg_h) * trg_h - src_h) // 2)
    pad_b = int((np.ceil(src_h / trg_h) * trg_h - src_h) - pad_t)
    pad_l = int((np.ceil(src_w / trg_w) * trg_w - src_w) // 2)
    pad_r = int((np.ceil(src_w / trg_w) * trg_w - src_w) - pad_l)
    img = F.pad(img.permute(2, 0, 1), [pad_l, pad_r, pad_t, pad_b])
    img = img.permute(1, 2, 0).numpy()
    
    src_h, src_w = img.shape[0], img.shape[1]
    patches = []
    num_h = (src_h - trg_h + stride) // stride
    num_w = (src_w - trg_w + stride) // stride
    for cnt_h in range(num_h):
        for cnt_w in range(num_w):
            patches.append(
                img[stride*cnt_h: stride*cnt_h+trg_h, stride*cnt_w: stride*cnt_w+trg_w, :] if len(img.shape)==3 \
                    else img[stride*cnt_h: stride*cnt_h+trg_h, stride*cnt_w: stride*cnt_w+trg_w]
            )
    return patches


def save_image(output, mean, norm, path2file):
    r"""
    Info:
        Save output to specific path.
    Args:
        - output (Tensor | ndarray): takes value from range [0, 1].
        - mean (float):
        - norm (float): 
        - path2file (str | os.PathLike):
    Returns:
        - (bool): indicate succeed or not.
    """
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    output = ((output.transpose((1, 2, 0)) * norm) + mean).astype(np.uint8)
    try:
        cv2.imwrite(path2file, output)
        return True
    except:
        return False


def visualize(array, folder, name="image.png", method="cv2"):
    r"""
    Info:
        Visualize the given array (RGB or GRAY).
    Args:
        - array (ndarray or Tensor): has shape (H, W) or (3, H, W) or (H, W, 3).
        - path (string): where to save image.
        - method (string): specify how to save image, "cv2" or "PIL"
    """
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    assert isinstance(array, np.ndarray), "Unsupported data format."
    assert 2 <= len(array.shape) <= 3, "Unsupported data shape {}.".format(array.shape)
    assert method in ["cv2", "PIL"], "Unsupported method {} for save image.".format(method)
    mode = "RGB" if len(array.shape) == 3 else "L"
    if mode == "RGB" and array.shape[0] == 3:
        array = np.transpose(array, axes=[1, 2, 0])
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, name)
    if method == "cv2":
        cv2.imwrite(path, array)
    else:
        Image.fromarray(array, mode=mode).save(path)


def resize(img: torch.Tensor, size: list or tuple, logger=None):
    r"""
    Info:
        Resize the input image. 
    Args:
        - img (torch.Tensor):
        - size (tuple | int): target size of image.
        - logger (Logger): record running information, if None, direct message to terminal.
    Returns:
        - img (torch.Tensor): image with target size. 
    """
    org_shape = img.shape
    if len(org_shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(org_shape) == 3:
        img = img.unsqueeze(0)
    elif len(org_shape) == 4:
        pass
    else:
        raise NotImplementedError("Function to deal with image with shape {} is not implememted yet.".format(org_shape))
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    img = img.reshape(org_shape)
    return img


def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    with log_info(msg="Set device for model.", level="INFO", state=True, logger=logger):
        if not torch.cuda.is_available():
            with log_info(msg="CUDA is not available, using CPU instead.", level="WARNING", state=False, logger=logger):
                device = torch.device("cpu")
        if len(gpu_list) == 0:
            with log_info(msg="Use CPU.", level="INFO", state=False, logger=logger):
                device = torch.device("cpu")
        elif len(gpu_list) == 1:
            with log_info(msg="Use GPU {}.".format(gpu_list[0]), level="INFO", state=False, logger=logger):
                device = torch.device("cuda:{}".format(gpu_list[0]))
                model = model.to(device)
        elif len(gpu_list) > 1:
            raise NotImplementedError("Multi-GPU mode is not implemented yet.")
    return model, device


def try_make_path_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return False
    return True


def save_ckpt(path2file, logger=None, **ckpt):
    with log_info(msg="Save checkpoint to {}".format(path2file), level="INFO", state=True, logger=logger):
        torch.save(ckpt, path2file)


def pack_code(cfg, logger=None):
    src_dir = cfg.GENERAL.ROOT
    src_items = [
        "src"
    ]
    des_dir = cfg.LOG.DIR
    with log_info(msg="Pack items {} from ROOT to {}".format(src_items, des_dir), level="INFO", state=True, logger=logger):
        t = time.localtime()
        for item in src_items:
            path2src = os.path.join(src_dir, item)
            path2des = os.path.join("{}/{}/Mon{}Day{}Hour{}Min{}".format(
                des_dir, 
                "src", 
                str(t.tm_mon).zfill(2), 
                str(t.tm_mday).zfill(2), 
                str(t.tm_hour).zfill(2), 
                str(t.tm_min).zfill(2), 
            ))
            try_make_path_exists(path2des)
            os.system("cp -r {} {}".format(path2src, path2des))
    # raise NotImplementedError("Function pack_code is not implemented yet.")



if __name__ == "__main__":
    log_info(msg="Hello", level="INFO", state=False, logger=None)

    