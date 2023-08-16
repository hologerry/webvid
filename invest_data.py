import argparse
import concurrent.futures
import os
import warnings

import numpy as np
import pandas as pd
import requests


def main(args):
    full_df = pd.read_csv(args.csv_path, sep=",")

    dog_df = full_df[full_df["name"].str.contains("dog")]
    cat_df = full_df[full_df["name"].str.contains("cat")]
    flower_df = full_df[full_df["name"].str.contains("flower")]
    tree_df = full_df[full_df["name"].str.contains("tree")]
    animal_df = full_df[full_df["name"].str.contains("animal")]

    print(f"dog: {len(dog_df)}")
    print(f"cat: {len(cat_df)}")
    print(f"flower: {len(flower_df)}")
    print(f"tree: {len(tree_df)}")
    print(f"animal: {len(animal_df)}")

    dog_out_path = args.csv_path.replace(".csv", "_dog.csv")
    cat_out_path = args.csv_path.replace(".csv", "_cat.csv")
    flower_out_path = args.csv_path.replace(".csv", "_flower.csv")
    tree_out_path = args.csv_path.replace(".csv", "_tree.csv")
    animal_out_path = args.csv_path.replace(".csv", "_animal.csv")

    dog_df.to_csv(dog_out_path, index=False)
    cat_df.to_csv(cat_out_path, index=False)
    flower_df.to_csv(flower_out_path, index=False)
    tree_df.to_csv(tree_out_path, index=False)
    animal_df.to_csv(animal_out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shutter Image/Video Downloader")

    parser.add_argument("--csv_path", type=str, default="../results_10M_val.csv", help="Path to csv data to download")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    main(args)

# 2M val
# dog: 29
# cat: 160
# flower: 89
# tree: 255
# animal: 18

# 2M train
# dog: 16104
# cat: 74926
# flower: 50848
# tree: 135962
# animal: 7856

# 10M val
# dog: 29
# cat: 158
# flower: 125
# tree: 281
# animal: 12

# 10M train
# dog: 69259
# cat: 324790
# flower: 219703
# tree: 587787
# animal: 33709
