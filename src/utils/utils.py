
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
from termcolor import colored


def notify(msg="", level="INFO", logger=None, fp=None):
    level = level.upper()
    if level == "WARNING":
        level = "[" + colored("{:<8}".format(level), "yellow") + "]"
    elif level == "ERROR":
        level = "[" + colored("{:<8}".format(level), "red")    + "]"
    elif level == "INFO":
        level = "[" + colored("{:<8}".format(level), "blue")   + "]"

    msg = "[{:<20}] {:<8} {}".format(time.asctime(), level, msg)
        
    _notify = print if logger is None else logger.log_info
    _notify(msg)
    
    if fp is None:
        return
    elif isinstance(fp, str):
        try:
            with open(fp, 'a') as _fp:
                _fp.write(msg)
        except:
            notify(msg="Failed to write message to file {}".format(fp), level="WARNING")
    else:
        try:
            fp.write(msg)
        except:
            notify(msg="Failed to write message to file.", level="WARNING")


@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    _state = "[" + colored("{:<8}".format("RUNNING"), "green") + "]" if state else ""
    notify(msg="{} {}".format(_state, msg), level=level, logger=logger)
    yield
    if state:
        _state = "[" + colored("{:<8}".format("DONE"), "green") + "]" if state else ""
        notify(msg="{} {}".format(_state, msg), level=level, logger=logger)


inform = notify


def raise_error(exc, msg):
    notify(msg=msg, level="ERROR")
    raise exc(msg)


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
            with log_info(msg=msg, level="INFO", state=True, logger=logger):
                res = func(*args, **kwargs)
            return res
        return wrapped_func
    return func_wraper


def resize_and_pad(img, resol, is_mask, to_tensor=True):
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
        raise_error(TypeError, "Unexpected img type {}".format(type(img)))
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
    padding_left = (resol[1] - img_resol[1]) // 2
    padding_right = resol[1] - img_resol[1] - padding_left
    padding_top = (resol[0] - img_resol[0]) // 2
    padding_bottom = resol[0] - img_resol[0] - padding_top
    img = F.pad(img, pad=(padding_left, padding_right, padding_top, padding_bottom))
    if dim == 3:
        img = img.squeeze(0)
    elif dim == 2:
        img = img.squeeze(0).squeeze(0)
    assert img.shape[-2: ] == resol, "Resolution error."
    assert len(img.shape) == dim, "Dimension inconsistent."
    if to_tensor:
        return img, (padding_left, padding_right, padding_top, padding_bottom)
    if org_type == "Ndarray":
        img = img.numpy()
    return img, (padding_left, padding_right, padding_top, padding_bottom)


def crop_and_resize(img, padding, resol, is_mask):
    r"""
    Info:
        Crop and resize image, the inverse of function resize_and_pad.
    Args:
        img (Ndarray | Tensor):
        padding (tuple of int): padding_left, padding_right, padding_top, padding_bottom
        resol (list | tuple): [H, W]
        is_mask (bool):
    Rets:
        img (Ndarray | Tensor): the type depends on input img's type.
    """
    if isinstance(img, np.ndarray):
        org_type = "Ndarray"
        img = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        org_type = "Tensor"
    else:
        raise_error(TypeError, "Unsupported image type {}".format(type(img)))
    org_dim = len(img.shape)
    if org_dim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif org_dim == 3:
        img = img.unsqueeze(0)

    img_h, img_w = resol
    padding_left, padding_right, padding_top, padding_bottom = padding
    org_h, org_w = img.shape[-2: ]

    img = img[..., padding_top: org_h-padding_bottom, padding_left: org_w-padding_right]
    img = F.interpolate(img.type(torch.float), size=(img_h, img_w), mode="nearest")
    assert img.shape[-2: ] == resol, "Shape of image inconsistent."
    
    if org_dim == 2:
        img = img.squeeze(0).squeeze(0)
    elif org_dim == 3:
        img = img.squeeze(0)

    # img = cv2.resize(img, (img_w, img_h), cv2.INTER_LINEAR)
    if is_mask:
        max_pix = torch.max(img)
        img = (img > (max_pix / 2)) * max_pix
    if org_type == "Ndarray":
        img = img.numpy()
        
    return img


def infer(model, data, device, infer_only=True, *args, **kwargs):
    r"""
    Info:
        Inference once, without calculate any loss.
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - device (torch.device)
        - infer_only (bool): if True, return the results not for calculate loss.
    Returns:
        - out (Tensor): predicted.
    """
    def _infer_V1(model, data, device, infer_only=True, *args, **kwargs):
        raise_error(NotImplementedError, "Not implemented yet.")

    return _inference_V1(model, data, device, infer_only, *args, **kwargs) 


def infer_and_calc_loss(model, data, loss_fn, device, *args, **kwargs):
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
        raise_error(NotImplementedError, "Not implemented yet")

    return _infer_and_calc_loss_V1(model, data, loss_fn, device, *args, **kwargs)


def calc_and_record_metrics(phase, epoch, outputs, targets, metrics_handler):
    batch_size = outputs.shape[0]
    for bs in range(batch_size):
        metrics_handler.update(phase, epoch, outputs[bs], targets[bs])


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
        raise_error(TypeError, "Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise_error(ValueError, "Input size must have a shape of (*, 3, H, W). Got {}"
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


def save_image(output, path2file):
    r"""
    Info:
        Save output to specific path.
    Args:
        - output (Tensor | ndarray): takes value from range [0, 1].
        - path2file (str | os.PathLike):
    Returns:
        - (bool): indicate succeed or not.
    """
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    # output = ((output.transpose((1, 2, 0)) * norm) + mean).astype(np.uint8)
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
        raise_error(NotImplementedError, "Function to deal with image with shape {} is not implememted yet.".format(org_shape))
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    # img = img.reshape(org_shape)
    return img


def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    # log_info = log_info if logger is None else logger.log_info
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
            raise_error(NotImplementedError, "Multi-GPU mode is not implemented yet.")
    return model, device


def set_pipline(model, cfg, logger=None):
    # log_info = log_info if logger is None else logger.log_info
    assert cfg.GENERAL.PIPLINE, "Not pipline model."
    gpu_list = cfg.GENERAL.GPU
    assert len(gpu_list) == 2, "Please specify 2 GPUs for pipline setting."
    with log_info(msg="Set pipline model.", level="INFO", state=True, logger=logger):
        device = torch.device("cuda:{}".format(gpu_list[0]))
        model.video_encoder.to(device)
        model.text_encoder.to(device)
        model.decoder.to(torch.device("cuda:{}".format(gpu_list[1])))
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



if __name__ == "__main__":
    log_info(msg="Hello", level="INFO", state=False, logger=None)

    