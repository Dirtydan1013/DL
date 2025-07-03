# src/compare_trimaps.py

import os
import argparse
import numpy as np
from PIL import Image
from oxford_pet import load_dataset

def compare_and_save(data_path, mode='test',
                     raw_out='raw_trimaps', proc_out='proc_masks'):
    """
    对照保存:
      1) 原始 trimap（三值：1=pet,2=border,3=bg）
      2) 经过 preprocess 后的二值 mask（0/1 → 0/255）
    文件名一一对应，便于人工比对。
    """
    os.makedirs(raw_out, exist_ok=True)
    os.makedirs(proc_out, exist_ok=True)

    # 载入 Dataset
    ds = load_dataset(data_path, mode=mode)
    for idx, fname in enumerate(ds.filenames):
        # --- a) 原始 trimap ---
        raw_path = os.path.join(data_path, 'annotations', 'trimaps', fname + '.png')
        raw_im = Image.open(raw_path)
        raw_im.save(os.path.join(raw_out, f"{fname}.png"))

        # --- b) 处理后的二值 mask ---
        sample = ds[idx]
        mask3d = sample['mask']            # numpy array, shape = (1, H, W)
        mask2d = np.squeeze(mask3d, axis=0)  # 变成 (H, W)
        arr = (mask2d * 255).astype(np.uint8)
        Image.fromarray(arr, mode='L')\
             .save(os.path.join(proc_out, f"{fname}.png"))

    print(f"Saved {len(ds.filenames)} pairs to:\n  {raw_out}/\n  {proc_out}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw trimap vs. processed GT mask"
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Dataset 根目录，包含 annotations/trimaps')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train','valid','test'],
                        help='选择拆分（train/valid/test）')
    parser.add_argument('--raw_out', type=str, default='raw_trimaps',
                        help='保存原始 trimap 的目录')
    parser.add_argument('--proc_out', type=str, default='proc_masks',
                        help='保存处理后 mask 的目录')
    args = parser.parse_args()

    compare_and_save(
        data_path=args.data_path,
        mode=args.mode,
        raw_out=args.raw_out,
        proc_out=args.proc_out
    )
