
r"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys, cv2, json, copy, h5py, math, random
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from utils import utils
# import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


class BERTTokenizer():
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self._build()

    def _build(self):
        self.s_text = "[CLS] "
        self.e_text = " [SEP]"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert_model.eval()

    def get_token(self, query):
        marked_text = self.s_text + query + self.e_text
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1]*len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segment_tensor = torch.tensor([segment_ids])
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, segment_tensor)
        hidden_states = outputs[2]
        token = torch.cat(hidden_states, dim=0).unsqueeze(0)
        token = token[..., 1: min(len(query), self.cfg.DATA.QUERY.MAX_WORDS) + 1, :]
        
        # Pad zeros.
        padding = max(0, self.cfg.DATA.QUERY.MAX_WORDS - token.shape[2])
        token = F.pad(token, pad=(0, 0, 0, padding))
        assert token.shape[2] == self.cfg.DATA.QUERY.MAX_WORDS, "Token shape error"
        # token = token[:, :, 0: self.cfg.DATA.QUERY.MAX_WORDS, :]
        token = token.squeeze(0).squeeze(0)
        return token

    def __call__(self, query):
        return self.get_token(query)


@add_dataset
class YRVOS2021FPS6(torch.utils.data.Dataset):
    def __init__(self, cfg, split, logger=None, *args, **kwargs):
        super(YRVOS2021FPS6, self).__init__()
        assert split in ["train", "valid"], "Unknown split "+split
        self.cfg = cfg
        self.split = split
        self.logger = logger
        self._build()

    def _build(self):
        self.items = []
        self.path2data = self.cfg.DATA.DIR.YoutubeRVOS2021FPS6
        self.num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        self.resol = self.cfg.DATA.VIDEO.RESOLUTION
        with open(os.path.join(self.path2data, "meta_expressions", self.split, "meta_expressions.json"), 'r') as fp:
            meta_expressions = json.load(fp)["videos"]
        with utils.log_info(msg="Load {} set.".format(self.split), state=True, logger=self.logger):
            for fn_video in meta_expressions.keys():
                if not os.path.exists(os.path.join(self.path2data, "HDF5Files", self.split, fn_video+".hdf5")):
                    continue
                fns = [fn.split(".")[0] for fn in  meta_expressions[fn_video]["frames"]]
                for expr_idx in meta_expressions[fn_video]["expressions"].keys():
                    # If video does not have enough frames
                    if len(meta_expressions[fn_video]["frames"]) < self.num_frames:
                        f_padding = (self.num_frames - len(meta_expressions[fn_video]["frames"])) // 2
                        b_padding = self.num_frames - len(meta_expressions[fn_video]["frames"]) - f_padding
                        self.items.append({
                            "expr_idx": expr_idx, 
                            "fn_video": fn_video, 
                            "query": meta_expressions[fn_video]["expressions"][expr_idx]["exp"], 
                            "fns": [fns[0]]*f_padding+fns+[fns[-1]]*b_padding, 
                        })
                        if self.split in ["train"]:
                            self.items[-1]["obj_idx"] = meta_expressions[fn_video]["expressions"][expr_idx]["obj_id"]
                        continue
                    # Else if video has enough frames
                    for cnt in range(len(meta_expressions[fn_video]["frames"])//self.num_frames):
                        self.items.append({
                            "expr_idx": expr_idx, 
                            "fn_video": fn_video, 
                            "query": meta_expressions[fn_video]["expressions"][expr_idx]["exp"], 
                            "fns": fns[cnt*self.num_frames: (cnt+1)*self.num_frames], 
                        })
                        if self.split in ["train"]:
                            self.items[-1]["obj_idx"] = meta_expressions[fn_video]["expressions"][expr_idx]["obj_id"]
                    if len(meta_expressions[fn_video]["frames"]) % self.num_frames != 0:
                        self.items.append({
                            "expr_idx": expr_idx, 
                            "fn_video": fn_video, 
                            "query": meta_expressions[fn_video]["expressions"][expr_idx]["exp"], 
                            "fns": fns[len(fns)-self.num_frames: len(fns)], 
                        })
                        if self.split in ["train"]:
                            self.items[-1]["obj_idx"] = meta_expressions[fn_video]["expressions"][expr_idx]["obj_id"]
        self.update()

    def __len__(self):
        # return len(self.items)
        return len(self.random_indexes)

    def update(self):
        if self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self.items))}
        else:
            k = int(len(self.items) * self.cfg.DATA.RANDOM_SAMPLE_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self.items)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}

    def __getitem__(self, idx):
        data = {}
        item = self.items[self.random_indexes[idx]]

        # Read HDF5File
        with h5py.File(os.path.join(self.path2data, "HDF5Files", self.split, item["fn_video"]+".hdf5")) as fp:
            frames = fp["video_5fps"][()]
            bert = fp["expr_{}/bert".format(item["expr_idx"])][()]
            fns = fp["fns"][()]
            if self.split in ["train"]:
                annos = fp["annotations"][()]
        assert bert.shape[-1] == self.cfg.DATA.QUERY.BERT_DIM, "Incorrect dimension of words."
        fns = [str(fn, encoding="utf-8") for fn in fns]

        # Read frames.
        start_frame_idx = fns.index(item["fns"][0])
        frames = frames[start_frame_idx: start_frame_idx+self.num_frames]
        f_padding = (self.num_frames - frames.shape[0]) // 2
        b_padding = self.num_frames - frames.shape[0] - f_padding
        frames = np.concatenate([frames[0][np.newaxis]]*f_padding+[frames]+[frames[-1][np.newaxis]]*b_padding, axis=0)
        frames = frames.astype(np.float)
        assert frames.shape[0] == self.num_frames, "Number of frames incorrect for video {}.".format(item["fn_video"])
        assert frames.shape[1] == 3, "Wrong format"
        assert len(item["fns"]) == self.num_frames, "Number of file names incorrect for video {}.".format(item["fn_video"])
        data["size"] = frames.shape[2: ]
        frames, padding = utils.resize_and_pad(frames, self.resol[0], is_mask=False)
        assert frames.shape[2: ] == self.resol[0], "Shape error"
        frames = (frames - 255.0/2) / (255.0/2)
        data["frames"] = frames

        # Read annotation.
        if self.split in ["train"]:
            annos = annos[start_frame_idx: start_frame_idx+self.num_frames]
            f_padding_annos = [copy.deepcopy(annos[0][None]) for i in range(f_padding)]
            b_padding_annos = [copy.deepcopy(annos[-1][None]) for i in range(b_padding)]
            annos = np.concatenate(f_padding_annos+[annos]+b_padding_annos, axis=0)
            annos = (annos == int(item["obj_idx"])).astype(np.float)
            assert annos.shape[0] == self.num_frames, "Number of annotations incorrect for video {}.".format(item["fn_video"])
            data["mask_l"], _ = utils.resize_and_pad(annos, self.resol[0], is_mask=True)
            data["mask_m"], _ = utils.resize_and_pad(annos, self.resol[1], is_mask=True)
            data["mask_s"], _ = utils.resize_and_pad(annos, self.resol[2], is_mask=True)

        # Read expression vectors.
        data["bert"] = bert

        # Other information.
        data["fns"] = item["fns"]
        data["fn_video"] = item["fn_video"]
        data["padding"] = padding
        
        return data


@add_dataset
class YoutubeRVOS2021FPS6(torch.utils.data.Dataset):
    def __init__(self, cfg, split, logger=None, *args, **kwargs):
        super(YoutubeRVOS2021FPS6, self).__init__()
        assert split in ["train", "valid"], "Unknown split "+split
        self.cfg = cfg
        self.split = split
        self.logger = logger
        self._build()

    def _build(self):
        self.items = []
        self.path2data = self.cfg.DATA.DIR.YoutubeRVOS2021FPS6
        self.num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        self.resol = self.cfg.DATA.VIDEO.RESOLUTION
        with open(os.path.join(self.path2data, "meta_expressions", self.split, "meta_expressions.json"), 'r') as fp:
            meta_expressions = json.load(fp)["videos"]
        with utils.log_info(msg="Load {} set.".format(self.split), state=True, logger=self.logger):
            for fn_video in meta_expressions.keys():
                if not os.path.exists(os.path.join(self.path2data, "HDF5Files", self.split, fn_video+".hdf5")):
                    continue
                for expr_idx in meta_expressions[fn_video]["expressions"].keys():
                    for frame_idx in meta_expressions[fn_video]["frames"]:
                        self.items.append({
                            "expr_idx": expr_idx, 
                            "fn_video": fn_video, 
                            "frame_idx": frame_idx.split(".")[0], 
                            "query": meta_expressions[fn_video]["expressions"][expr_idx]["exp"], 
                        })
                        if self.split in ["train"]:
                            self.items[-1]["obj_idx"] = meta_expressions[fn_video]["expressions"][expr_idx]["obj_id"]
        self.update()

    def __len__(self):
        # return len(self.items)
        return len(self.random_indexes)

    def update(self):
        if self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self.items))}
        else:
            k = int(len(self.items) * self.cfg.DATA.RANDOM_SAMPLE_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self.items)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}

    def __getitem__(self, idx):
        data = {}
        item = self.items[self.random_indexes[idx]]

        # Read HDF5File
        with h5py.File(os.path.join(self.path2data, "HDF5Files", self.split, item["fn_video"]+".hdf5")) as fp:
            frames = fp["video_5fps"][()]
            bert = fp["expr_{}/bert".format(item["expr_idx"])][()]
            fns = fp["fns"][()]
            if self.split in ["train"]:
                annos = fp["annotations"][()]

        fns = [str(fn, encoding="utf-8") for fn in fns]

        # Read frames.
        front_padding = self.num_frames // 2 - fns.index(item["frame_idx"])
        front_padding = max(front_padding, 0)
        back_padding = (self.num_frames // 2 - 1) - (len(fns) - 1 - fns.index(item["frame_idx"]))
        back_padding = max(back_padding, 0)
        first_read_frame_idx = (fns.index(item["frame_idx"]) - (self.num_frames // 2)) if front_padding == 0 else 0
        last_read_frame_idx = fns.index(item["frame_idx"]) + (self.num_frames // 2 - 1) if back_padding == 0 else (len(fns) - 1)

        frames = np.concatenate(
            [frames[0][None]]*front_padding + [frames[first_read_frame_idx: last_read_frame_idx+1]] + [frames[-1][None]]*back_padding, 
        axis=0).astype(np.float)
        assert frames.shape[0] == self.num_frames, "Number of frames error"
        assert frames.shape[1] == 3, "Wrong format"
        data["size"] = frames.shape[2: ]
        frames, padding = utils.resize_and_pad(frames, self.resol[0], is_mask=False)
        frames = (frames - 255.0/2) / (255.0/2)
        data["frames"] = frames

        # Read annotation.
        if self.split in ["train"]:
            annos = np.concatenate(
                [annos[0][None]]*front_padding + [annos[first_read_frame_idx: last_read_frame_idx]] + [annos[-1][None]]*back_padding, 
            axis=0)
            assert annos.shape[0] == self.num_frames, "Number of frames error"
            annos = (annos == int(item["obj_idx"])).astype(np.float)
            # data["mask"] = annos
            data["mask_l"], _ = utils.resize_and_pad(annos, self.resol[0], is_mask=True)
            data["mask_m"], _ = utils.resize_and_pad(annos, self.resol[1], is_mask=True)
            data["mask_s"], _ = utils.resize_and_pad(annos, self.resol[2], is_mask=True)

        # Read expression vectors.
        data["bert"] = bert

        # Other information.
        data["frame_idx"] = item["frame_idx"]
        data["fn_video"] = item["fn_video"]
        
        return data


@add_dataset
class YoutubeRVOS2021(torch.utils.data.Dataset):
    def __init__(self, cfg, split, logger=None, *args, **kwargs):
        super(YoutubeRVOS2021, self).__init__()
        assert split in ["train", "valid"], "Unknown split "+split
        self.cfg = cfg
        self.split = split
        self.logger = logger
        self._build()

    def _build(self):
        self.bert_tokenizer = BERTTokenizer(self.cfg)
        self.items = []
        self.path2data = self.cfg.DATA.DIR.YoutubeRVOS2021
        self.num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        self.resol = self.cfg.DATA.VIDEO.RESOLUTION
        self.path2videos = os.path.join(self.path2data, "all_frames", self.split+"_all_frames", "JPEGImages")
        with open(os.path.join(self.path2data, "meta_expressions", self.split, "meta_expressions.json"), 'r') as fp:
            meta_expressions = json.load(fp)["videos"]
        with utils.log_info(msg="Load {} set.".format(self.split), state=True, logger=self.logger):
            for fn_video in meta_expressions.keys():
                video_fns = sorted(os.listdir(os.path.join(self.path2videos, fn_video)))
                first_frame_idx = video_fns[0].split(".")[0]
                last_frame_idx = video_fns[-1].split(".")[0]
                for expr_idx in meta_expressions[fn_video]["expressions"].keys():
                    for frame_idx in meta_expressions[fn_video]["frames"]:
                        self.items.append({
                            "fn_video": fn_video, 
                            "frame_idx": frame_idx.split(".")[0], 
                            "query": meta_expressions[fn_video]["expressions"][expr_idx]["exp"], 
                            "first_frame_idx": first_frame_idx, 
                            "last_frame_idx": last_frame_idx, 
                        })
                        if self.split in ["train"]:
                            self.items[-1]["obj_idx"] = meta_expressions[fn_video]["expressions"][expr_idx]["obj_id"]
        self.update()

    def __len__(self):
        # return len(self.items)
        return len(self.random_indexes)

    def update(self):
        if self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self.items))}
        else:
            k = int(len(self.items) * self.cfg.DATA.RANDOM_SAMPLE_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self.items)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}

    def __getitem__(self, idx):
        data = {}
        item = self.items[self.random_indexes[idx]]

        # Read frames.
        front_padding = self.num_frames // 2 - (int(item["frame_idx"]) - int(item["first_frame_idx"]))
        front_padding = max(front_padding, 0)
        back_padding = (self.num_frames // 2 - 1) - (int(item["last_frame_idx"]) - int(item["frame_idx"]))
        back_padding = max(back_padding, 0)
        first_read_frame_idx = (int(item["frame_idx"]) - (self.num_frames // 2)) if front_padding == 0 else int(item["first_frame_idx"])
        last_read_frame_idx = int(item["frame_idx"]) + (self.num_frames // 2 - 1) if back_padding == 0 else int(item["last_frame_idx"])
        frames = []
        for idx in range(first_read_frame_idx, last_read_frame_idx+1):
            img = cv2.imread(os.path.join(self.path2videos, item["fn_video"], str(idx).zfill(5)+".jpg"), -1)
            assert img is not None, "Expect frame {} of video {} is a RGB image, but got None".format(str(idx).zfill(5), item["fn_video"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis]
            frames.append(img.transpose((0, 3, 1, 2)))
        frames = np.concatenate([frames[0]]*front_padding+frames+[frames[-1]]*back_padding, axis=0)
        assert frames.shape[0] == self.num_frames, "Number of frames error"
        assert frames.shape[1] == 3, "Wrong format"
        data["size"] = frames.shape[-2: ]
        frames, padding = utils.resize_and_pad(frames, self.resol[0], is_mask=False)
        frames = (frames - 255.0/2) / (255.0/2)
        data["frames"] = frames

        # Read annotation.
        if self.split in ["train"]:
            anno = np.array(Image.open(os.path.join(self.path2data, self.split, "Annotations", item["fn_video"], item["frame_idx"]+".png")))
            anno = (anno == int(item["obj_idx"])).astype(np.float)
            # data["mask"] = anno
            data["mask_l"], _ = utils.resize_and_pad(anno, self.resol[0], is_mask=True)
            data["mask_m"], _ = utils.resize_and_pad(anno, self.resol[1], is_mask=True)
            data["mask_s"], _ = utils.resize_and_pad(anno, self.resol[2], is_mask=True)

        # Read expression vectors.
        data["bert"] = self.bert_tokenizer(item["query"])

        # Other information.
        data["fn_video"] = item["fn_video"]
        data["frame_idx"] = item["frame_idx"]
        data["padding"] = padding

        return data


@add_dataset
class A2DSentences(torch.utils.data.Dataset):
    def __init__(self, cfg, split, logger=None, *args, **kwargs):
        super(A2DSentences, self).__init__()
        assert split in ["train", "valid"], "Unknown split"
        self.cfg = cfg
        self.split = split
        self.logger = logger
        self._build()

    def _build(self):
        """
        The path to file `train.txt` and `test.txt` contains following data.
        `video name`, `frame id`, `instance`, `height`, `width`, `category`, `query`
        """
        self.resol = self.cfg.DATA.VIDEO.RESOLUTION
        _file_path = os.path.join(self.cfg.DATA.DIR.A2DSentences, self.split+".csv")
        self.bert_tokenizer = BERTTokenizer(self.cfg)
        with utils.log_info(msg="Load {} set.".format(self.split), state=True, logger=self.logger):
            self._item_list = []
            with open(_file_path, 'r') as f:
                content = f.readlines()
            self._item_list = [{
                "video": item.split(', ')[0], 
                # "frame": item.split(', ')[1], # start from 00001
                "frame": item.split(', ')[1].zfill(5), # start from 00001
                "instance": item.split(', ')[2], 
                "height": item.split(', ')[3], 
                "width": item.split(', ')[4], 
                "category": item.split(', ')[5], 
                "query": item.split(', ')[6].split('\n')[0], 
            } for item in content]
        self.update()

    def __len__(self):
        return len(self.random_indexes)

    def update(self):
        if self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self._item_list))}
        else:
            k = int(len(self._item_list) * self.cfg.DATA.RANDOM_SAMPLE_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self._item_list)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        data = {}
        item = self._item_list[self.random_indexes[index]]
        item["frame"] = str(int(item["frame"])).zfill(5)
        path_to_item = os.path.join(self.cfg.DATA.DIR.A2DSentences, self.split, item["video"]+".hdf5")
        with h5py.File(path_to_item, 'r', libver="latest", swmr=True) as f:
            frames = f["video"][()]
            data["size"] = np.array([int(item["height"]), int(item["width"])])
            mask = f["{}/masks/{}/mask".format(item["frame"], item["instance"])][()].astype(np.float32)
            data["query"] = item["query"]
            # data["text"] = f["{}/text".format(item["instance"])][()].squeeze(0)
            if item["instance"] not in f.keys() or "bert" not in f[item["instance"]].keys():
               data["bert"] = self.bert_tokenizer(item["query"])
            else:
                # data["text"] = f["{}/text".format(item["instance"])][()].squeeze(0)
                # data["bert"] = torch.tensor(f["{}/bert".format(item["instance"])][()]).unsqueeze(0)
                data["bert"] = self.bert_tokenizer(data["query"])
            data["obj_idx"] = item["instance"]
            data["frame_idx"] = item["frame"]
            # data["category"] = item["category"]
            data["fn_video"] = item["video"]
        # NOTE I3D model data pre-processing.
        num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        front_padding = num_frames // 2 - int(data["frame_idx"]) + 1 if int(data["frame_idx"]) - 1 < num_frames // 2 else 0
        back_padding = (num_frames - num_frames // 2 - 1) - (frames.shape[0] - int(data["frame_idx"])) if (frames.shape[0] - int(data["frame_idx"])) < (num_frames - num_frames // 2 - 1) else 0
        start = int(data["frame_idx"]) - 1 - num_frames // 2 if int(data["frame_idx"]) > num_frames // 2 else 0
        end = int(data["frame_idx"]) - 1 + num_frames - num_frames // 2 if int(data["frame_idx"]) + num_frames - num_frames // 2 -1 <= frames.shape[0] else frames.shape[0] - 1
        # frames = torch.from_numpy(frames)
        frames = np.concatenate([frames[0][None]]*front_padding + [frames[start: end]] + [frames[-1][None]]*back_padding)
        frames = frames.astype(np.float32)
        assert frames.shape[0] == num_frames, \
            "Number of frames error, fetching frame {} of video {}, start: {}, end: {}, front pading: {}, back padding{}, expect {} but got {}.".format(
                data["frame_idx"], data["fn_video"], start, end, front_padding, back_padding, num_frames, frames.shape[0]
            )
        resolution = self.cfg.DATA.VIDEO.RESOLUTION
        
        # Read frames.
        frames = torch.from_numpy(frames)
        frames, padding = utils.resize_and_pad(frames, self.resol[0], is_mask=False)
        frames = (frames - 255.0/2) / (255.0/2)
        data["frames"] = frames

        # Read annotations.
        data["mask_l"], _ = utils.resize_and_pad(mask, self.resol[0], is_mask=True)
        data["mask_m"], _ = utils.resize_and_pad(mask, self.resol[1], is_mask=True)
        data["mask_s"], _ = utils.resize_and_pad(mask, self.resol[2], is_mask=True)
        
        data["bbox_l"] = utils.get_coord(data["mask_l"])
        data["bbox_m"] = utils.get_coord(data["mask_m"])
        data["bbox_s"] = utils.get_coord(data["mask_s"])

        # Other information.
        data["padding"] = padding

        # data["raw_clip"] = frames[num_frames//2, :, :, :]
        return data


@add_dataset
class JHMDBSentences(torch.utils.data.Dataset):
    def __init__(self, cfg, split, logger=None, *args, **kwargs):
        super(JHMDBSentences, self).__init__()
        assert split in ["train", "valid"], "Unknown split"
        self.cfg = cfg
        self.split = split
        self.logger = logger
        self._build()

    def _build(self):
        """
        The path to file `train.txt` and `test.txt` contains following data.
        `video name`, `frame id`, `instance`, `height`, `width`, `category`, `query`
        """
        self.resol = self.cfg.DATA.VIDEO.RESOLUTION
        _file_path = os.path.join(self.cfg.DATA.DIR.JHMDBSentences, "raw", self.split+".csv")
        self.bert_tokenizer = BERTTokenizer(self.cfg)
        with utils.log_info(msg="Load {} set.".format(self.split), state=True, logger=self.logger):
            self._item_list = []
            with open(_file_path, 'r') as f:
                content = f.readlines()
            self._item_list = [{
                "video": item.split(', ')[0], 
                # "frame": item.split(', ')[1], # start from 00001
                "frame": item.split(', ')[1].zfill(5), # start from 00001
                "instance": item.split(', ')[2], 
                "height": item.split(', ')[3], 
                "width": item.split(', ')[4], 
                "category": item.split(', ')[5], 
                "query": item.split(', ')[6].split('\n')[0], 
            } for item in content]
        self.update()

    def __len__(self):
        return len(self.random_indexes)

    def update(self):
        if self.split in ["valid", "test"]:
            self.random_indexes = {idx: idx for idx in range(len(self._item_list))}
        else:
            k = int(len(self._item_list) * self.cfg.DATA.RANDOM_SAMPLE_RATIO)
            random_samples_idx = sorted(random.sample(range(0, len(self._item_list)), k))
            self.random_indexes = {idx: random_samples_idx[idx] for idx in range(k)}

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        data = {}
        item = self._item_list[self.random_indexes[index]]
        item["frame"] = str(int(item["frame"])).zfill(5)
        path_to_item = os.path.join(self.cfg.DATA.DIR, self.split, item["video"]+".hdf5")
        with h5py.File(path_to_item, 'r', libver="latest", swmr=True) as f:
            frames = f["video"][()]
            data["size"] = np.array([int(item["height"]), int(item["width"])])
            mask = f["{}/masks/{}/mask".format(item["frame"], item["instance"])][()].astype(np.float32)
            data["query"] = item["query"]
            # data["text"] = f["{}/text".format(item["instance"])][()].squeeze(0)
            if item["instance"] not in f.keys() or "bert" not in f[item["instance"]].keys():
               data["bert"] = self.bert_tokenizer(item["query"])
            else:
                # data["text"] = f["{}/text".format(item["instance"])][()].squeeze(0)
                data["bert"] = torch.tensor(f["{}/bert".format(item["instance"])][()]).unsqueeze(0)
            data["obj_idx"] = item["instance"]
            data["frame_idx"] = item["frame"]
            # data["category"] = item["category"]
            data["fn_video"] = item["video"]
        # NOTE I3D model data pre-processing.
        num_frames = self.cfg.DATA.VIDEO.NUM_FRAMES
        front_padding = num_frames // 2 - int(data["frame_idx"]) + 1 if int(data["frame_idx"]) - 1 < num_frames // 2 else 0
        back_padding = (num_frames - num_frames // 2 - 1) - (frames.shape[0] - int(data["frame_idx"])) if (frames.shape[0] - int(data["frame_idx"])) < (num_frames - num_frames // 2 - 1) else 0
        start = int(data["frame_idx"]) - 1 - num_frames // 2 if int(data["frame_idx"]) > num_frames // 2 else 0
        end = int(data["frame_idx"]) - 1 + num_frames - num_frames // 2 if int(data["frame_idx"]) + num_frames - num_frames // 2 -1 <= frames.shape[0] else frames.shape[0] - 1
        # frames = torch.from_numpy(frames)
        frames = np.concatenate([frames[0][None]]*front_padding + [frames[start: end]] + [frames[-1][None]]*back_padding)
        frames = frames.astype(np.float32)
        assert frames.shape[0] == num_frames, \
            "Number of frames error, fetching frame {} of video {}, start: {}, end: {}, front pading: {}, back padding{}, expect {} but got {}.".format(
                data["frame_idx"], data["fn_video"], start, end, front_padding, back_padding, num_frames, frames.shape[0]
            )
        resolution = self.cfg.DATA.VIDEO.RESOLUTION
        
        # Read frames.
        frames = torch.from_numpy(frames)
        frames, padding = utils.resize_and_pad(frames, self.resol[0], is_mask=False)
        frames = (frames - 255.0/2) / (255.0/2)
        data["frames"] = frames

        # Read annotations.
        data["mask_l"], _ = utils.resize_and_pad(mask, self.resol[0], is_mask=True)
        data["mask_m"], _ = utils.resize_and_pad(mask, self.resol[1], is_mask=True)
        data["mask_s"], _ = utils.resize_and_pad(mask, self.resol[2], is_mask=True)

        data["bbox_l"] = utils.get_coord(data["mask_l"])
        data["bbox_m"] = utils.get_coord(data["mask_m"])
        data["bbox_s"] = utils.get_coord(data["mask_s"])

        # Other information.
        data["padding"] = padding

        # data["raw_clip"] = frames[num_frames//2, :, :, :]
        return data


@add_dataset
class Dataset(torch.utils.data.Dataset):
    r"""
    Info:
        This is a demo dataset.
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __len__(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset is not implemeted yet.")


if __name__ == "__main__":
    from configs.configs import cfg
    dataset = YRVOS2021FPS6(cfg, "train")
    # dataset = YoutubeRVOS2021(cfg, "train")
    # dataset = A2DSentences(cfg, "train")
    # for idx in range(len(dataset)):
    #     if len(dataset.items[idx]["fns"]) != cfg.DATA.VIDEO.NUM_FRAMES:
    #         print(dataset.items[idx]["fn_video"])
    #         print(dataset.items[idx]["fns"])
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4
    )
    # pbar = tqdm(total=len(data_loader))
    pbar = tqdm(total=len(dataset))
    # dataset[326]
    for idx, data in enumerate(dataset):
        # if idx < 5:
        #     print(data["bert"].shape)
        pbar.update()





