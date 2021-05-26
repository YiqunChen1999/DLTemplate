
r"""
Author:
    Yiqun Chen
Docs:
    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils, loss_fn_helper, lr_scheduler_helper, optimizer_helper
from utils.logger import Logger
from utils.metrics import MetricsHandler
from models import model_builder
from data import data_loader
from train import train_one_epoch
from evaluate import evaluate
from inference import inference

def main():
    # Set logger to record information.
    utils.check_env(cfg)
    logger = Logger(cfg)
    logger.log_info(cfg)
    metrics_handler = MetricsHandler(cfg.metrics)
    # utils.pack_code(cfg, logger=logger)

    # Build model.
    model = model_builder.build_model(cfg=cfg, logger=logger)
    optimizer = optimizer_helper.build_optimizer(cfg=cfg, model=model)
    lr_scheduler = lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)

    # Read checkpoint.
    ckpt = torch.load(cfg.model.path2ckpt) if cfg.gnrl.resume else {}
    if cfg.gnrl.resume:
        with logger.log_info(msg="Load pre-trained model.", level="INFO", state=True, logger=logger):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            
    # Set device.
    model, device = utils.set_pipline(model, cfg) if cfg.gnrl.PIPLINE else utils.set_device(model, cfg.gnrl.cuda)
    
    resume_epoch = ckpt["epoch"] if cfg.gnrl.resume else 0
    loss_fn = loss_fn_helper.build_loss_fn(cfg=cfg)
    
    # Prepare dataset.
    train_loaders, valid_loaders, test_loaders = dict(), dict(), dict()
    for dataset in cfg.data.datasets:
        if cfg.data[dataset].TRAIN:
            try:
                train_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "train")
            except:
                utils.notify(msg="Failed to build train loader of %s" % dataset)
        if cfg.data[dataset].VALID:
            try:
                valid_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "valid")
            except:
                utils.notify(msg="Failed to build valid loader of %s" % dataset)
        if cfg.data[dataset].TEST:
            try:
                test_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "test")
            except:
                utils.notify(msg="Failed to build test loader of %s" % dataset)

    
    # TODO Train, evaluate model and save checkpoint.
    for epoch in range(cfg.train.max_epoch):
        epoch += 1
        if resume_epoch >= epoch:
            continue
        
        eval_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model, "loss_fn": loss_fn, "device": device, 
            "metrics_handler": metrics_handler, "logger": logger, "save": cfg.save.save, 
        }
        train_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model, "loss_fn": loss_fn, "optimizer": optimizer, 
            "device": device, "lr_scheduler": lr_scheduler, "metrics_handler": metrics_handler, "logger": logger, 
        }
        ckpt_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model.state_dict(), "metrics_handler": metrics_handler, 
            "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), 
        }

        for dataset in cfg.data.datasets:
            if cfg.data[dataset].TRAIN:
                utils.notify("Train on %s" % dataset)
                train_one_epoch(data_loader=train_loaders[dataset], **train_kwargs)

        utils.save_ckpt(path2file=cfg.model.path2ckpt, **ckpt_kwargs)

        if epoch in cfg.gnrl.ckphs:
            utils.save_ckpt(path2file=os.path.join(cfg.model.ckpts, cfg.gnrl.id + "_" + str(epoch).zfill(5) + ".pth"), **ckpt_kwargs)
            for dataset in cfg.data.datasets:
                utils.notify("Evaluating test set of %s" % dataset, logger=logger)
                if cfg.data[dataset].TEST:
                    evaluate(data_loader=test_loaders[dataset], phase="test", **eval_kwargs)

        for dataset in cfg.data.datasets:
            utils.notify("Evaluating valid set of %s" % dataset, logger=logger)
            if cfg.data[dataset].VALID:
                evaluate(data_loader=valid_loaders[dataset], phase="valid", **eval_kwargs)
    # End of train-valid for loop.
    
    eval_kwargs = {
        "epoch": epoch, "cfg": cfg, "model": model, "loss_fn": loss_fn, "device": device, 
        "metrics_handler": metrics_handler, "logger": logger, "save": cfg.save.save, 
    }

    for dataset in cfg.data.datasets:
        if cfg.data[dataset].VALID:
            utils.notify("Evaluating valid set of %s" % dataset, logger=logger)
            evaluate(data_loader=valid_loaders[dataset], phase="valid", **eval_kwargs)
    for dataset in cfg.data.datasets:
        if cfg.data[dataset].TEST:
            utils.notify("Evaluating test set of %s" % dataset, logger=logger)
            evaluate(data_loader=test_loaders[dataset], phase="test", **eval_kwargs)

    for dataset in cfg.data.datasets:
        if "train" in cfg.data[dataset].INFER:
            utils.notify("Inference on train set of %s" % dataset)
            inference(data_loader=train_loaders[dataset], phase="infer_train", **eval_kwargs)
        if "valid" in cfg.data[dataset].INFER:
            utils.notify("Inference on valid set of %s" % dataset)
            inference(data_loader=valid_loaders[dataset], phase="infer_valid", **eval_kwargs)
        if "test" in cfg.data[dataset].INFER:
            utils.notify("Inference on test set of %s" % dataset)
            inference(data_loader=test_loaders[dataset], phase="infer_test",  **eval_kwargs)

    return None


if __name__ == "__main__":
    main()
