import os

from path import Path
from pose_utils import gen_poses
import sys
import argparse
parser = argparse.ArgumentParser()

def imgs2poses(work_dir):
    num_used_image = gen_poses(work_dir, 'exhaustive_matcher')
    return num_used_image

if __name__ == "__main__":
    gen_poses(os.path.join(Path('porf_data/dtu'), "gun6"), 'exhaustive_matcher')