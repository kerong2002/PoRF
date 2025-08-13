import os

from path import Path
from pose_utils import gen_poses
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='porf_data/dtu/gun6', help='Path to the dataset directory.')

def imgs2poses(work_dir):
    num_used_image = gen_poses(work_dir, 'exhaustive_matcher')
    return num_used_image

if __name__ == "__main__":
    args = parser.parse_args()
    imgs2poses(args.work_dir)