
r"""
Info:
    Read arguments from terminal.
Author:
    Yiqun Chen
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id",                             type=str,   required=True)
parser.add_argument("--batch_size",                     type=int,   required=True)
parser.add_argument("--lr",         default=1e-4,       type=float, required=True)
parser.add_argument("--max_epoch",  default=20,         type=int,   required=True)
parser.add_argument("--resume",     default="false",    type=str,   required=True,  choices=["true", "false"])
parser.add_argument("--cuda",                           type=str,   required=True)
args = parser.parse_args()