
r"""
Author  Yiqun Chen
Docs    Main functition to run program.
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
    logger = Logger(cfg)
    logger.log_info(cfg)
    metrics_handler = MetricsHandler(cfg.METRICS)
    utils.pack_code(cfg, logger=logger)

    # Build model.
    model = model_builder.build_model(cfg=cfg, logger=logger)
    optimizer = optimizer_helper.build_optimizer(cfg=cfg, model=model)
    lr_scheduler = lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)

    # Read checkpoint.
    ckpt = torch.load(cfg.MODEL.PATH2CKPT) if cfg.GENERAL.RESUME else {}
    if cfg.GENERAL.RESUME:
        with utils.log_info(msg="Load pre-trained model.", level="INFO", state=True, logger=logger):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            
    # Set device.
    model, device = utils.set_pipline(model, cfg) if cfg.GENERAL.PIPLINE else utils.set_device(model, cfg.GENERAL.GPU)
    
    resume_epoch = ckpt["epoch"] if cfg.GENERAL.RESUME else 0
    loss_fn = loss_fn_helper.build_loss_fn(cfg=cfg)
    
    # Prepare dataset.
    train_loaders, valid_loaders, test_loaders = dict(), dict(), dict()
    for dataset in cfg.DATA.DATASETS:
        try:
            train_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "train")
        except:
            utils.notify(msg="Failed to build train loader of %s" % dataset)
        try:
            valid_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "valid")
        except:
            utils.notify(msg="Failed to build valid loader of %s" % dataset)
        try:
            test_loaders[dataset] = data_loader.build_data_loader(cfg, dataset, "test")
        except:
            utils.notify(msg="Failed to build test loader of %s" % dataset)

    
    # TODO Train, evaluate model and save checkpoint.
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        epoch += 1
        if resume_epoch >= epoch:
            continue
        
        eval_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model, "device": device, 
            "metrics_handler": metrics_handler, "logger": logger, "save": cfg.SAVE.SAVE, 
        }
        train_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model, "loss_fn": loss_fn, "optimizer": optimizer, 
            "device": device, "lr_scheduler": lr_scheduler, "metrics_handler": metrics_handler, "logger": logger, 
        }
        ckpt_kwargs = {
            "epoch": epoch, "cfg": cfg, "model": model.state_dict(), "metrics_handler": metrics_handler, 
            "optimizer": optimizer.state_dict(), , "lr_scheduler": lr_scheduler, 
        }

        for dataset in cfg.DATA.DATASETS:
            if cfg.DATA[dataset].TRAIN:
                utils.notify("Train on %s" % dataset)
                train_one_epoch(data_loader=train_data_loader, **train_kwargs)

        utils.save_ckpt(path2file=cfg.MODEL.PATH2CKPT, **ckpt_kwargs)

        if epoch in cfg.GENERAL.CHECK_EPOCHS:
            utils.save_ckpt(path2file=os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + "_" + str(epoch).zfill(5) + ".pth"), **ckpt_kwargs)
            for dataset in cfg.DATA.DATASETS:
                utils.notify("Evaluating test set of %s" % dataset, logger=logger)
                if cfg.DATA[dataset].TEST:
                    evaluate(data_loader=test_data_loader, phase="test", **eval_kwargs)

        for dataset in cfg.DATA.DATASETS:
            utils.notify("Evaluating valid set of %s" % dataset, logger=logger)
            if cfg.DATA[dataset].VALID:
                evaluate(data_loader=valid_data_loader, phase="valid", **eval_kwargs)
    # End of train-valid for loop.
    
    eval_kwargs = {
        "epoch": epoch, "cfg": cfg, "model": model, "device": device, 
        "metrics_handler": metrics_handler, "logger": logger, "save": cfg.SAVE.SAVE, 
    }

    for dataset in cfg.DATA.DATASETS:
        if cfg.DATA[dataset].VALID:
            utils.notify("Evaluating valid set of %s" % dataset, logger=logger)
            evaluate(data_loader=valid_data_loader, phase="valid")
    for dataset in cfg.DATA.DATASETS:
        if cfg.DATA[dataset].TEST:
            utils.notify("Evaluating test set of %s" % dataset, logger=logger)
            evaluate(data_loader=test_data_loader, phase="test")

    for dataset in cfg.DATA.DATASETS:
        if "train" in cfg.DATA[dataset].INFER:
            utils.notify("Inference on train set of %s" % dataset)
            inference(data_loader=train_loaders[dataset], phase="infer_train", **eval_kwargs)
        if "valid" in cfg.DATA[dataset].INFER:
            utils.notify("Inference on valid set of %s" % dataset)
            inference(data_loader=valid_loaders[dataset], phase="infer_valid", **eval_kwargs)
        if "test" in cfg.DATA[dataset].INFER:
            utils.notify("Inference on test set of %s" % dataset)
            inference(data_loader=test_loaders[dataset], phase="infer_test",  **eval_kwargs)

    return None


if __name__ == "__main__":
    main()
