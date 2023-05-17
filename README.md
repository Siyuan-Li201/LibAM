# LibAM
This repository contains the datasets and source code for the paper, ["LibAM: An Area Matching Framework for Detecting Third-party Libraries in Binaries"](https://arxiv.org/abs/2305.04026v2).
![application1](https://github.com/Siyuan-Li201/LibAM/assets/128788605/151e58ad-af60-4dda-9e44-8cc831e78f4f)

# Environment
- Python 3.8 
- IDA Pro 6.8 (with Python 2.7 package)

#### The `requirements.txt` file for Python dependencies is located in the `envs/` directory.

# Structure
- `code/`: All source code of LibAM ()
- `dataset/`: All four datasets used in LibAM
- `envs/`: Python dependencies of LibAM
- `groundtruth/`: The ground truth for dataset2 and dataset3 in LibAM
- `code/libam/data/`: All detection results and intermediate data are saved in the directory
- [`result/`](https://drive.google.com/drive/folders/1XWvBt0CfocXbayrAwHAkeHUt3ruXLs_a?usp=sharing): Final detection results in paper and all intermediate data from the detection process. You can use this data rather than raw binaries in `dataset/` to run any process of LibAM.

(For the purpose of large-scale analysis, the code is set to run with multiple processes by default. However, it can be easily modified to run with a single process. In fact, for dataset2 and dataset3, running with a single process is already quite fast.)

# Run
## 1. Prepare dataset
#### Copy `dataset2` or `dataset3` from `dataset/` to `code/libam/data/`. (The target binaries in dataset are not striped for manully analyzing the detection results of LibAM, which is same as in previous paper. Note that removing the function names in target binaries doesn't affect LibAM detection)
#### Modify the `DATA_PATH` in `code/libam/settings.py` to select a dataset to run (both the data path and tool path in `settings.py` should be set).
#### Note: You only need to prepare IDA Pro and Python 2.7 if you want to run from raw binaries in the `dataset/`. You can decompress the [`result/`](https://drive.google.com/drive/folders/1XWvBt0CfocXbayrAwHAkeHUt3ruXLs_a?usp=sharing) and place it instead of `dataset/` into `code/libam/data/data/`, and use the pre-extracted features provided by us.

## 2. Preprocess binaries
#### Run `python 1_preprocess.py`
#### Use IDA Pro to extract ACFG and FCG from binaries.
#### Note: You should only run this step if you want to extract features by yourself.

## 3. Embed the functions and FCGs
#### Run `python 2_embedding.py`

## 4. Compare functions
#### Run `python 3_func_compare.py`
#### Use Annoy to accelerate the vector search phase.

## 5. Run TPL detection task
#### Run `python 4_tpl_detection.py`
#### Execute the Embedded-GNN and Anchor Alignment Algorithm in LibAM to generate TPL detection results.

## 6. Run Area detection task
#### Run `python 5_area_detection.py`
#### Generate the reuse areas and calculate the scores for area detection tasks.
