from glob import glob
import cv2
import numpy as np
import pandas as pd
import polars as pl
import random
from sklearn.model_selection import StratifiedGroupKFold

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def frameid_lag_checker(start_frame, end_frame,stack_startpoint,video_name, start_lag=0):
    stack_dice = []
    for video_frame_idx in range(start_frame, end_frame+1):
        stack_dice.append({"frame_id": video_frame_idx, "frame_offset": start_lag, "actual_frame_id": video_frame_idx - start_lag, "video":video_name})
        if video_frame_idx == stack_startpoint:
            stack_startpoint +=6
            start_lag +=1
    return pl.DataFrame(stack_dice)

seed_everything()

all_list = []
labels = [0, 5,11,12,13,21,22,23,24,25,31,32,33,50, 255]

for d1 in glob(f"../../data/CholecSeg8k/**/"):
    for file in glob(f"{d1}/**/*_watershed_mask.png"):
        file_dict = {"video": d1.split("/")[-2], "file": file, "frame_id": int(file.split("/")[-1].split("_")[1]), "subgroup": file.split("/")[-2]}
        a = cv2.imread(file)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        for lbl in labels:
            file_dict[lbl] = (a == lbl).sum() > 0
        for i in np.unique(a):
            if i not in labels:
                print(f"----{file} has label {i} ")
        all_list.append(file_dict)

df = pd.DataFrame(all_list)
# video09 - 5, video01 - 23, 24, 25, video12 - 23, 25, 33
# video09: [5]00832-1071 
# video01 [24]14859-15178, 16345-16504, 16585-16664, 28580-28979 (692 frames) [23]14859-15178, 16345-16504, 16585-16664, 28580-28979 [25]16345-16504, 16585-16664
# video12  [33] 15750-15910, 19500-19576, 19900-19980, [25] 15851, 15856, [23] 15750-15909, 19500-19820, 19980-20059

# Fixed test clips
# video01_16585, video01_28580, video01_28660, video01_28740, video01_28820, video01_28900, video09_00992, video12_19900, video12_19980
# Other clips from video01, video09, video12 are fixed train
video01 = df[df['video'] == 'video01'].drop(columns=[0]).sort_values(by="frame_id")
video09 = df[df['video'] == 'video09'].drop(columns=[0]).sort_values(by="frame_id")
video12 = df[df['video'] == 'video12'].drop(columns=[0]).sort_values(by="frame_id")

included_test = []
for test_clip in ['video01_16585', 'video01_28580', 'video01_28660', 'video01_28740', 'video01_28820', 'video01_28900', 'video09_00992', 'video12_19900', 'video12_19980', 'video12_19500']:
    tmp = df[df['subgroup'] == test_clip]
    included_test.append(tmp)

included_test = pd.concat(included_test)
included_train = pd.concat([video01, video09, video12])
included_train = included_train[~included_train['subgroup'].isin(
    ['video01_16585', 'video01_28580', 'video01_28660', 'video01_28740', 'video01_28820', 'video01_28900', 'video09_00992', 'video12_19900', 'video12_19980', 'video12_19500']
    )]

# 5-fold split on remaining clips

excluded = df[~df['video'].isin(['video01', 'video09', 'video12'])].drop(columns=[0]) 
excluded.groupby("video").sum()
excluded['stratify'] = excluded[32] + excluded[13]*2

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

DF_LI = pl.read_csv("cholecseg8k_lag_correction.csv").to_dicts()
correct_df_li = []
for _set in DF_LI:
    correct_df_li.append(frameid_lag_checker(_set["start_frame"], _set["end_frame"], _set["stack_startpoint"], _set["video_name"]))
correct_df = pl.concat(correct_df_li)

for i, (train_index, test_index) in enumerate(skf.split(excluded, excluded['stratify'], groups=excluded['video'])):
    excluded_train = excluded.iloc[train_index]
    excluded_test = excluded.iloc[test_index]

    train = pd.concat([included_train.drop(columns=["subgroup"]), excluded_train.drop(columns=["stratify", "subgroup"])]).reset_index(drop=True)
    test = pd.concat([included_test.drop(columns=["subgroup"]), excluded_test.drop(columns=["stratify","subgroup"])]).reset_index(drop=True)

    train = pl.from_pandas(train)
    test = pl.from_pandas(test)

    train = train.join(correct_df, on=["frame_id", "video"], how="left").fill_null(0).with_columns(
        (pl.col("frame_id") - pl.col("frame_offset").cast(pl.Int64)).alias("actual_frame_id")
    )
    test = test.join(correct_df, on=["frame_id", "video"], how="left").fill_null(0).with_columns(
        (pl.col("frame_id") - pl.col("frame_offset").cast(pl.Int64)).alias("actual_frame_id")
    )

    train.write_csv(f"splits/cholecseg8k_train_{i}.csv")
    test.write_csv(f"splits/cholecseg8k_test_{i}.csv")