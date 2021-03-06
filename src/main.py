
r"""
Author  Yiqun Chen
Docs    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils, loss_fn_helper, lr_scheduler_helper, optimizer_helper
from utils.logger import Logger
from utils.metrics import Metrics
from models import model_builder
from data import data_loader
from train import train_one_epoch
from evaluate import evaluate
from inference import inference

def main():
    # TODO Set logger to record information.
    # Set logger to record information.
    logger = Logger(cfg)
    logger.log_info(cfg)
    metrics_logger = Metrics()
    utils.pack_code(cfg, logger=logger)

    # Build model.
    model = model_builder.build_model(cfg=cfg, logger=logger)

    # Read checkpoint.
    ckpt = torch.load(cfg.MODEL.PATH2CKPT) if cfg.GENERAL.RESUME else {}

    if cfg.GENERAL.RESUME:
        with utils.log_info(msg="Load pre-trained model.", level="INFO", state=True, logger=logger):
            model.load_state_dict(ckpt["model"])

            # Set device.
            if cfg.GENERAL.PIPLINE:
                model, device = utils.set_pipline(model, cfg)
            else:
                model, device = utils.set_device(model, cfg.GENERAL.GPU)
            
            optimizer = optimizer_helper.build_optimizer(cfg=cfg, model=model)
            optimizer.load_state_dict(ckpt["optimizer"])
            
            lr_scheduler = lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)
            # lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    else:
        # Set device.
        if cfg.GENERAL.PIPLINE:
            model, device = utils.set_pipline(model, cfg)
        else:
            model, device = utils.set_device(model, cfg.GENERAL.GPU)
        optimizer = optimizer_helper.build_optimizer(cfg=cfg, model=model)
        lr_scheduler = lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)

    resume_epoch = ckpt["epoch"] if cfg.GENERAL.RESUME else 0
    # loss_fn = ckpt["loss_fn"] if cfg.GENERAL.RESUME else loss_fn_helper.build_loss_fn(cfg=cfg)
    loss_fn = loss_fn_helper.build_loss_fn(cfg=cfg)
    
    # Prepare dataset.
    try:
        train_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "train")
    except:
        logger.log_info("Can not build data loader for train set.")
    try:
        valid_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "valid")
    except:
        logger.log_info("Can not build data loader for valid set.")
    try:
        test_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "test")
    except:
        logger.log_info("Can not build data loader for test set.")

    # ################ NOTE DEBUG NOTE ################
    with utils.log_info(msg="Debug", level="INFO", state=True, logger=logger):
        '''train_one_epoch(
            epoch=0,
            cfg=cfg,  
            model=model, 
            data_loader=train_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            metrics_logger=metrics_logger, 
            logger=logger, 
        )
        evaluate(
            epoch=0, 
            cfg=cfg, 
            model=model, 
            data_loader=valid_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            metrics_logger=metrics_logger, 
            phase="valid", 
            logger=logger,
            save=cfg.SAVE.SAVE,  
        )
        inference(
            cfg=cfg, 
            model=model, 
            data_loader=train_data_loader, 
            device=device, 
            phase="train", 
            logger=logger, 
        )'''
    # ################ NOTE DEBUG NOTE ################
    
    # TODO Train, evaluate model and save checkpoint.
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        if not cfg.GENERAL.TRAIN:
            break
        if resume_epoch >= epoch:
            continue
        if cfg.GENERAL.TRAIN:
            train_one_epoch(
                epoch=epoch,
                cfg=cfg,  
                model=model, 
                data_loader=train_data_loader, 
                device=device, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                metrics_logger=metrics_logger, 
                logger=logger, 
            )

        optimizer.zero_grad()
        with torch.no_grad():
            utils.save_ckpt(
                path2file=cfg.MODEL.PATH2CKPT, 
                cfg=cfg, 
                logger=logger, 
                model=model.state_dict(), 
                epoch=epoch, 
                optimizer=optimizer.state_dict(), 
                lr_scheduler=lr_scheduler.state_dict(), # NOTE Need attribdict>=0.0.5
                loss_fn=loss_fn, 
                # metrics=metrics_logger, 
            )
        if epoch in cfg.SCHEDULER.UPDATE_EPOCH:
            with torch.no_grad():
                utils.save_ckpt(
                    path2file=os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + "_" + str(epoch).zfill(3) + ".pth"), 
                    cfg=cfg, 
                    logger=logger, 
                    model=model.state_dict(), 
                    epoch=epoch, 
                    optimizer=optimizer.state_dict(), 
                    lr_scheduler=lr_scheduler.state_dict(), # NOTE Need attribdict>=0.0.5
                    loss_fn=loss_fn, 
                )
            if cfg.GENERAL.TEST:
                evaluate(
                    epoch=epoch, 
                    cfg=cfg, 
                    model=model, 
                    data_loader=test_data_loader, 
                    device=device, 
                    loss_fn=loss_fn, 
                    metrics_logger=metrics_logger, 
                    phase="test", 
                    logger=logger,
                    save=cfg.SAVE.SAVE,  
                )
        if cfg.GENERAL.VALID:
            evaluate(
                epoch=epoch, 
                cfg=cfg, 
                model=model, 
                data_loader=valid_data_loader, 
                device=device, 
                loss_fn=loss_fn, 
                metrics_logger=metrics_logger, 
                phase="valid", 
                logger=logger,
                save=cfg.SAVE.SAVE,  
            )
    # End of train-valid for loop.

    if cfg.GENERAL.VALID:
        evaluate(
            epoch=epoch, 
            cfg=cfg, 
            model=model, 
            data_loader=valid_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            metrics_logger=metrics_logger, 
            phase="valid", 
            logger=logger,
            save=cfg.SAVE.SAVE,  
        )
    if cfg.GENERAL.TEST:
        evaluate(
            epoch=epoch, 
            model=model, 
            data_loader=test_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            logger=logger,
            save=cfg.SAVE.SAVE,  
        )
    
    if cfg.GENERAL.INFER != "none":
        if cfg.GENERAL.INFER == "train":
            infer_data_loader = train_data_loader
        elif cfg.GENERAL.INFER == "valid":
            infer_data_loader = valid_data_loader
        elif cfg.GENERAL.INFER == "test":
            infer_data_loader = test_data_loader
        else:
            raise ValueError("Expect dataset for inference is one of ['train', 'valid', 'test'], but got {}".format(cfg.GENERAL.INFER))
        evaluate(
            epoch=epoch, 
            cfg=cfg, 
            model=model, 
            data_loader=infer_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            metrics_logger=metrics_logger, 
            phase="valid", 
            logger=logger,
            save=cfg.SAVE.SAVE,  
        )
    return None


if __name__ == "__main__":
    main()
