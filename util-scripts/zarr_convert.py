import os
import argparse
import glob
import numpy as np
import h5py, zarr
from numcodecs import Blosc
from tqdm import tqdm

def main(data_dir, out_dir):
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    os.makedirs(f"{out_dir}", exist_ok=True)

    for file in tqdm(glob.glob(f"{data_dir}/*.h5"), desc="Converting files..."):
        img_id = os.path.splitext(os.path.split(file)[1])[0]

        with h5py.File(file, "r") as h5f:
            z = zarr.open(f"hsi_61_zarr/{img_id}.zarr", mode="w")
            key = "cube"
            data = np.array(h5f[key], dtype=np.float32)
            data = data.transpose(2, 0, 1)
            z.create_dataset(
                "data",
                shape=data.shape,
                chunks=(61, 512, 512),
                dtype=data.dtype,
                compressor=compressor,
            )[:] = data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="HSI dataset to convert (directory with h5 files)")
    parser.add_argument("--out_dir", "-o", type=str, required=False, default="./hsi_61_zarr", help="output directory")
    args = parser.parse_args()
    main(args.dataset, args.out_dir)

