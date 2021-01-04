
"""
Author  Yiqun Chen
Docs    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils, loss_fn_helper, lr_scheduler_helper, optimizer_helper
from utils.logger import Logger
from models import model_builder
from data import data_loader
from train import train_one_epoch
from evaluate import evaluate

def main():
    # TODO Read configuration.
    # TODO Set logger to record information.
    # TODO Build model.
    # TODO Read checkpoint.
    # TODO Load pre-trained model.
    # TODO Set device.
    # TODO Prepare dataset.
    # TODO Train, evaluate model and save checkpoint.
    raise NotImplementedError("Function main do not implemented yet.")


if __name__ == "__main__":
    main()
