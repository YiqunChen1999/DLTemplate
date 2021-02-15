
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

def cal_ssim_pt(im1, im2, data_range=None, multichannel=True, *args, **kwargs):
    assert im1.shape == im2.shape, "Shapes of im1 and im2 are not equal."
    device = kwargs.pop("device", torch.device("cpu"))
    if len(im1.shape) == 3:
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
    if im1.shape[-1] == 3:
        im1 = im1.permute(0, 3, 1, 2)
        im2 = im2.permute(0, 3, 1, 2)
    channels = 3 if multichannel else 1
    win_size = kwargs.pop("win_size", 7)
    num_pixels = win_size ** 2
    mean_1 = F.conv2d(im1, torch.ones(channels, 1, win_size, win_size, device=device) / num_pixels, groups=channels)
    mean_2 = F.conv2d(im2, torch.ones(channels, 1, win_size, win_size, device=device) / num_pixels, groups=channels)
    var_1 = F.conv2d((im1 ** 2), torch.ones(channels, 1, win_size, win_size, device=device) / num_pixels, groups=channels) - mean_1 ** 2
    var_2 = F.conv2d((im2 ** 2), torch.ones(channels, 1, win_size, win_size, device=device) / num_pixels, groups=channels) - mean_2 ** 2
    covar = F.conv2d((im1 * im2), torch.ones(channels, 1, win_size, win_size, device=device) / num_pixels, groups=channels) - mean_1 * mean_2
    K1 = kwargs.pop("K1", 0.01)
    K2 = kwargs.pop("K2", 0.03)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    ssim = ( (2 * mean_1 * mean_2 + C1) * (2 * covar + C2) ) \
        / ( (mean_1 ** 2 + mean_2 ** 2 + C1) * (var_1 + var_2 + C2) )
    mssim = ssim.mean()
    return mssim

def inference(model, data, device):
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
    def _inference_V1(model, data, device):
        src = data["src"]
        src = src.to(device)
        out = model(src)
        return out, 

    raise NotImplementedError("Function utils.inference is not implemented yet.")
    return _inference_V1(model, data, device) 

def inference_and_calc_loss(model, data, loss_fn, device):
    r"""
    Info:
        Execute inference and calculate loss, sychronize the train and evaluate progress. 
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - loss_fn (callable): function or callable instance.
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
        - loss (Tensor): calculated loss.
    """
    def _infer_and_calc_loss_V1(model, data, loss_fn, device):
        # NOTE Only work with _inference_V1
        out, *_ = inference(model, data, device)
        trg = data["trg"].to(device)
        loss = loss_fn(out, trg)
        return out, loss

    raise NotImplementedError("Function utils.inference_and_calc_loss is not implemented yet.")
    return _infer_and_calc_loss_V1(model, data, loss_fn, device)

def cal_and_record_metrics(phase, epoch, output, target, metrics_logger, logger=None):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    batch_size = output.shape[0]
    for idx in range(batch_size):
        metrics_logger.cal_metrics(phase, epoch, target[idx], output[idx], data_range=1)

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
    from skimage import color
    from skimage import data
    img = data.astronaut()
    np_img_hsv = color.rgb2hsv(img).transpose((2, 0, 1))
    np_img_hsv[0] = np_img_hsv[0] * 2 * math.pi
    pt_img_hsv = rgb2hsv(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)/1.0)
    print("The End.")
    '''vec1 = torch.randn((2, 3, 5, 5)) * (-1)
    vec2 = torch.randn((2, 3, 5, 5)) * 2

    assert vec1.shape == vec2.shape, "ShapeError"
    assert vec1.shape[1] == 3, "ShapeError"
    numerator = torch.sum(vec1.permute(0, 2, 3, 1) * vec2.permute(0, 2, 3, 1), dim=3, keepdim=True)
    denominator = torch.sqrt(
        torch.sum(vec1.permute(0, 2, 3, 1) ** 2, dim=3, keepdim=True) \
            * torch.sum(vec2.permute(0, 2, 3, 1) ** 2, dim=3, keepdim=True)
    )
    cosine = numerator / denominator
    print(cosine)'''

    