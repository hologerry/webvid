import argparse
import concurrent.futures
import os
import warnings

import numpy as np
import pandas as pd
import requests

from tqdm import tqdm


def video_to_frames(video_path, frame_path):
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    os.system(f"ffmpeg -y -i {video_path} -hide_banner -loglevel error -q:v 1 {frame_path}")


def request_save(url, video_save_path, frame_save_path):
    img_data = requests.get(url, timeout=5).content
    with open(video_save_path, "wb") as handler:
        handler.write(img_data)

    video_to_frames(video_save_path, frame_save_path)


def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, "videos")
    frames_dir = os.path.join(args.data_dir, "frames")
    os.makedirs(video_dir, exist_ok=True)

    # COMM.barrier()

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace(".csv", f"_{args.partitions}")

    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, sub_df in enumerate(df_split):
            sub_df.to_csv(os.path.join(partition_dir, f"{idx}.csv"), index=False)

    relevant_fp = os.path.join(args.data_dir, "relevant_videos_exists.txt")
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(os.path.join(args.data_dir, "relevant_videos_exists.txt"), names=["fn"])
    else:
        exists = []

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    # data_dir/results_csvsplit/results_0.csv
    # data_dir/results_csvsplit/results_1.csv
    # ...
    # data_dir/results_csvsplit/results_N.csv

    df = pd.read_csv(os.path.join(partition_dir, f"{args.part}.csv"))

    df["rel_fn"] = df.apply(lambda x: os.path.join(str(x["page_dir"]), str(x["videoid"])), axis=1)

    df["rel_fn"] = df["rel_fn"] + ".mp4"

    df = df[~df["rel_fn"].isin(exists)]

    # remove nan
    df.dropna(subset=["page_dir"], inplace=True)

    playlists_to_dl = np.sort(df["page_dir"].unique())

    desc = f"Part {args.part} of {args.partitions}"

    for page_dir in tqdm(playlists_to_dl, desc=desc):
        vid_dir_t = os.path.join(video_dir, page_dir)
        frame_dir_t = os.path.join(frames_dir, page_dir)
        pdf = df[df["page_dir"] == page_dir]
        if len(pdf) > 0:
            os.makedirs(vid_dir_t, exist_ok=True)

            urls_todo = []
            video_save_paths = []
            frames_save_paths = []

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row["videoid"]) + ".mp4")
                frame_fp = os.path.join(frame_dir_t, str(row["videoid"]), "%06d.jpg")
                if not os.path.isfile(video_fp):
                    urls_todo.append(row["contentUrl"])
                    video_save_paths.append(video_fp)
                    frames_save_paths.append(frame_fp)

            print(f"Spawning {len(urls_todo)} jobs for page {page_dir}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {
                    executor.submit(request_save, url, vid_fp, frame_fp)
                    for url, vid_fp, frame_fp in zip(urls_todo, video_save_paths, frames_save_paths)
                }
            # request_save(urls_todo[0], video_save_paths[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shutter Image/Video Downloader")
    parser.add_argument(
        "--partitions",
        type=int,
        default=4,
        help="Number of partitions to split the dataset into, to run multiple jobs in parallel",
    )
    parser.add_argument(
        "--part", type=int, required=True, help="Partition number to download where 0 <= part < partitions"
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where webvid data is stored.")
    parser.add_argument("--csv_path", type=str, default="results_2M_train.csv", help="Path to csv data to download")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    # if SIZE > 1:
    #     warnings.warn("Overriding --part with MPI rank number")
    #     args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)
