import os
import sys
from pathlib import Path
import numpy as np
import shutil
import time
from functools import partial
import multiprocessing
import argparse

def call_blender_proc(id_obj, list_cad_path, list_output_dir, disable_output):
    output_dir = list_output_dir[id_obj]
    cad_path = list_cad_path[id_obj]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(os.path.join(output_dir, 'color_000041.png')):
        return
    
    command = "blenderproc run data_utils/blenderproc.py {} {}".format(cad_path, output_dir)
    if disable_output:
        command += " true"
    else:
        command += " false"
    os.system(command)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render objects using BlenderProc')
    parser.add_argument('--start', type=int, default=0, help='Start index for ID slice')
    parser.add_argument('--end', type=int, default=10_000, help='End index for ID slice')
    parser.add_argument('--cad_root', type=str, default='data/objaverse/hf-objaverse-v1/glbs', help='Path to CAD models')
    parser.add_argument('--output_root', type=str, default='data/objaverse_renderings', help='Path to save renderings')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)
    
    names = open('data/10k.txt').read().splitlines()[args.start:args.end]
    output_dirs = [os.path.join(args.output_root, name.split('.')[0]) for name in names]
    cad_paths = [os.path.join(args.cad_root, name + '.glb') for name in names]
    
    call_blender_proc_with_index = partial(call_blender_proc, list_cad_path=cad_paths, list_output_dir=output_dirs, disable_output=True)
    pool = multiprocessing.Pool(processes=min(args.end - args.start, 10))
    list(pool.imap_unordered(call_blender_proc_with_index, range(len(names))))
