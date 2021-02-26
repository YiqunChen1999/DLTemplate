
r"""
Author:
    Yiqun Chen
Docs:
    Orginaze data.
"""

import os, sys, json, cv2, h5py
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

def orginaze_youtubevos2021_6fps():
    def orginaze_queries():
        raise NotImplementedError("Method orginaze_queries is not implemented yet.")

    def orginaze_videos():
        raise NotImplementedError("Method orginaze_videos is not implemented yet.")

    def orginaze_train_files():
        split = "train"
        from configs.configs import cfg
        from data.dataset import BERTTokenizer
        bert_tokenizer = BERTTokenizer(cfg)
        path2video = "/data/YoutubeRVOS2021/{}/JPEGImages".format(split)
        path2annos = "/data/YoutubeRVOS2021/{}/Annotations".format(split)
        hdf5_folder = "/data/YoutubeRVOS2021/HDF5Files/{}".format(split)
        with open(os.path.join("/data/YoutubeRVOS2021/meta_expressions/{}/meta_expressions.json".format(split)), 'r') as fp:
            meta_expressions = json.load(fp)["videos"]
        pbar = tqdm(total=len(meta_expressions.keys()))
        for fn_video in meta_expressions.keys():
            if not os.path.exists(os.path.join(path2video, fn_video)):
                print(fn_video)
                continue
            fns = sorted(os.listdir(os.path.join(path2video, fn_video)))
            frames = []
            annos = []
            path2hdf5 = os.path.join(hdf5_folder, "{}.hdf5".format(fn_video))
            hdf5_exist = os.path.exists(path2hdf5)
            if not hdf5_exist:
                for fn in fns:
                    path2frame = os.path.join(path2video, fn_video, fn)
                    img = cv2.imread(path2frame, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis]
                    img = np.transpose(img, axes=(0, 3, 1, 2))
                    frames.append(img)

                    # annos.append(np.array(Image.open(os.path.join(path2annos, fn_video, fn.replace(".jpg", ".png"))))[np.newaxis])
                frames = np.concatenate(frames, axis=0)
                # annos = np.concatenate(annos, axis=0)
            fns = [fn.split(".")[0] for fn in fns]
            with h5py.File(path2hdf5, 'r+') as fp:
                fp.create_dataset("fns", data=fns)
                # if not hdf5_exist:
                    # fp.create_dataset("video_5fps", data=frames, compression="gzip")
                    # fp.create_dataset("annotations", data=annos, compression="gzip")
                # for expr_idx, expr_item in meta_expressions[fn_video]["expressions"].items():
                    # fp.create_dataset("expr_{}/expr".format(expr_idx), data=expr_item["exp"])
                    # fp.create_dataset("expr_{}/obj_idx".format(expr_idx), data=expr_item["obj_id"])
                    # fp.create_dataset("expr_{}/bert".format(expr_idx), data=bert_tokenizer(expr_item["exp"]), compression="gzip")
            pbar.update()
    orginaze_train_files()


if __name__ == "__main__":
    # import json
    # with open("/data/YoutubeRVOS2021/meta_expressions/train/meta_expressions.json", 'r') as fp:
    #     train_json = json.load(fp)
    # print(type(train_json))

    orginaze_youtubevos2021_6fps()

