# Aria-NeRF
This is the GitHub repository for the toolset of the Aria-Dataset. The Aria-Dataset is open for ongoing collection. If you are interested in contributing, please let us know.

## Setup

#### Create environment

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name aria_nerf -y python=3.8
conda activate aria_nerf
python -m pip install --upgrade pip
conda activate aria_nerf
```

### Aria Tools
Following the Aria_data_tool [installation instruction](https://github.com/facebookresearch/Aria_data_tools/blob/main/BUILD.md), we use `projectaria_tools==1.0.0`
```bash
pip install projectaria_tools==1.0.0
```
For other dependencies, please run.
```bash
pip install tqdm numpy plotly matplotlib opencv-contrib-python
```

If you meet the issue `AttributeError: 'tqdm_notebook' object has no attribute 'disp'` with `tqdm`, please run the following command:
```bash
pip install ipywidgets
```

### nerfstudio (Optional)

If you need to use the `COLMAP` algorithm or `nerfacto` baseline in `nerfstudio`, please refer to the [installation guide for nerfstudio](https://docs.nerf.studio/quickstart/installation.html).

### NeuralDiff (Optional)
If you need to run the `NeuralDiff` baseline, please refer to the installation guide for [NeuralDiff](https://github.com/dichotomies/NeuralDiff).

## Dataset
Download the dataset [here](https://office365stanford-my.sharepoint.com/:f:/r/personal/jksun_stanford_edu/Documents/Public_Dataset/Aria_NeRF_Dataset?csf=1&web=1&e=95670D). The file structure is as follows:
```bash
SCENE_NAME
    ├── closed_loop_trajectory.csv
    ├── description.txt
    ├── generalized_eye_gaze.csv
    ├── global_points.csv.gz
    ├── online_calibration.jsonl
    ├── open_loop_trajectory.csv
    ├── open_loop_trajectory.euroc
    ├── ${SCENE_NAME}.json
    ├── ${SCENE_NAME}.vrs
    ├── semidense_observations.csv.gz
    └── summary.json
```
Audio file `.wav` can be extracted from the `.vrs` file. 

First, extract the RGB files from the `.vrs` file and save to `${RGB}` folder for preprocessing.
```bash
python mps_extract_data.py
```
Next, use the `COLMAP` algorithm from `nerfstudio` to estimate camera positions for images in `${RGB}` and save to `${RGB_Pose}`.
```bash
# process RGB images
ns-process-data images --data ${RGB} --output-dir ${RGB_Pose}
```
## Training

The following will train a `nerfacto` model, which is recommended for real world scenes.
```bash
# Train model using ${RGB_Pose} data (nerfstudio is required)
ns-train nerfacto --data ${RGB_Pose}
# Eval and report metrics using the config.yml path given by nerfstudio
ns-eval --load-config ${config_path}
# Render
ns-render interpolate --load-config ${config_path}
```

The following will train a `NeuralDiff` model, a dynamic NeRF method.
```bash
cd ${NeuralDiff_DIR}
# Convert for dataset for NeuralDiff
python -m colmap_converter --colmap_dir ${RGB_Pose}
mkdir -p data/EPIC-Diff/
cd data/EPIC-Diff
ln -s ../custom/colmap/ P01_01
cd P01_01
ln -s images frames
cd ../../..
sh scripts/train.sh P01_01
rm -rf results/rel/P01_01/summary
sh scripts/eval.sh rel P01_01 rel 'summary' 0 0
sh scripts/eval.sh rel P01_01 rel 'masks' 0 0
```

## Reference
**[Aria-NeRF: Multimodal Egocentric View Synthesis
](https://arxiv.org/abs/2311.06455)**
<br />
[Jiankai Sun](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en), 
[Jianing Qiu](), 
[Chuanyang Zheng](),
[John Tucker](), 
[Javier Yu](), and
[Mac Schwager]()
<br />
[[Paper]](https://arxiv.org/abs/2311.06455)

```
@misc{sun2023arianerf,
      title={Aria-NeRF: Multimodal Egocentric View Synthesis}, 
      author={Jiankai Sun and Jianing Qiu and Chuanyang Zheng and John Tucker and Javier Yu and Mac Schwager},
      year={2023},
      eprint={2311.06455},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
- [Aria_data_tools](https://github.com/facebookresearch/Aria_data_tools)
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [NeuralDiff](https://github.com/dichotomies/NeuralDiff)