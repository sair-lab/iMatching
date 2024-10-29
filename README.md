<h1 align="center">iMatching: Imperative Correspondence Learning</h1>

<p align="center"><strong>
    <a href = "https://www.linkedin.com/in/zitong-zhan-134a85141/">Zitong Zhan</a><sup>*</sup>,
    <a href = "https://scholar.google.com/citations?user=_loctXsAAAAJ">Dasong Gao</a><sup>*</sup>,
    Yun-Jou Lin, Youjie Xia,
    <a href = "https://sairlab.org/team/chenw/">Chen Wang</a>,
</strong></p>


<p align="center"><strong>
    <a href = "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5586_ECCV_2024_paper.php">&#128196; [ECCV 2024]</a> | 
    <a href = "https://arxiv.org/abs/2312.02141">&#128196; [Arxiv]</a> | 
    <a href = "https://sairlab.org/iMatching/">&#128190; [Project Site]</a>
</strong></p>


# Dependencies

The following library/packages are required:

 - Python >= 3.10
 - PyTorch >= 2.1
 - [GTSAM](https://gtsam.org/) >= 4.1.1
 - [`requirement.txt`](requirements.txt)

## Steps

0. Optionally, prepare build env

   ```sh
   conda create -n imatching python=3.10
   conda activate imatching
   conda install gcc==9.5 gxx==9.5 cmake boost
   ```

1. Follow the official [PyTorch instructions](https://pytorch.org/get-started/locally/) to install PyTorch >= 1.11.
    
    ```sh
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    ```

2. Install the rest of Python dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Install GTSAM with Python bindings:

    ```sh
    # install gtsam dependencies
    conda install boost==1.82.0

    # build gtsam from source
    export BUILD_TYPE=RelWithDebInfo
    git submodule update --init --recursive thirdparty/gtsam
    # cmake build
    cd thirdparty/gtsam

    mkdir -p build
    cd build
    # TBB allocator crashes python so we have to turn it off
    cmake .. -DGTSAM_BUILD_PYTHON=1 -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DGTSAM_WITH_TBB=OFF -DGTSAM_CMAKE_CXX_FLAGS=-march-native
    make -j python-install
    ```

> **Note**: In case `libgtsamRelWithDebInfo.so` is not found, run `sudo ldconfig` to register the GTSAM library. 

> **Note**: If you are on Python 3.11 and the above procedure throws compile error about pybind11, you may need to replace `thirdparty/gtsam/wrap/pybind11` with [pybind11 >= 2.10.1](https://github.com/pybind/pybind11/releases) (tested on 2.10.4). 


4. Update submodules
    ```
    git submodule update --init --recursive ext/
    ```

# Data
## TartanAir
 - [TartanAir](https://theairlab.org/tartanair-dataset/), download with [tartanair_tools](https://github.com/castacks/tartanair_tools)
```
 python download_training.py --output-dir OUTPUTDIR --rgb --depth --only-left
```
The dataset is aranged as follows:
```
$DATASET_ROOT/
└── tartanair/
    ├── abandonedfactory_night/
    |   ├── Easy/
    |   |   └── ...
    │   └── Hard/
    │       └── ...
    └── ...
```

> **Note**: For TartanAir, only `<ENVIRONMENT>/<DIFFICULTY>/<image|depth>_left.zip` is required. After `unzip`ing downloaded zip files, make sure to remove the duplicate `<ENVIRONMENT>` directory level (`tartanair/abandonedfactory/abandonedfactory/Easy/...` -> `tartanair/abandonedfactory/Easy/...`).
```
find . -type f -name '*.zip' -exec unzip -n {} \;
for i in abandonedfactory        amusement   endofworld  hospital       neighborhood  office   oldtown      seasonsforest         soulcity abandonedfactory_night  carwelding  gascola     japanesealley  ocean         office2  seasidetown  seasonsforest_winter  westerndesert
do
    rm -r $i/Easy
    rm -r $i/Hard
    mv $i/$i/* $i/
    rm -r $i/$i
done
```
- [ETH3D SLAM](https://www.eth3d.net/slam_datasets): 
The dataset will be downloaded automatically by the datamodule.
# Weights
```sh
pip install gdown
mkdir pretrained
cd pretrained/
# CAPS
gdown 1UVjtuhTDmlvvVuUlEq_M5oJVImQl6z1f
#p2p
sh ../ext/patch2pix/pretrained/download.sh
#aspan
gdown 1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k
tar -xvf weights_aspanformer.tar
cd ..
```

# Run

## Train on TartanAir
- CAPS
```sh
scene=abandonedfactory
d=Easy
python ./train.py \
data_root=./data/datasets \
datamodule.include=\"$scene\_$d\"
```
- Patch2Pix
```sh
scene=abandonedfactory
d=Easy
python ./train.py \
--config-name p2p-train-tartanair \
data_root=./data/datasets \
trainer.max_epochs=2 \
datamodule.include=\"$scene\_$d\"
```
- AspanFormer
```sh
scene=abandonedfactory
d=Easy
python ./train.py \
--config-name aspan-train-tartanair \
data_root=./data/datasets \
trainer.max_epochs=2 \
datamodule.include=\"$scene\_$d\"
```

- DKM
```sh
scene=abandonedfactory
d=Easy
python ./train.py \
--config-name dkm-train-tartanair \
data_root=./data/datasets \
trainer.max_epochs=5 \
datamodule.include=\"$scene\_$d\"
```

## Override configs

Configs listed in `python ./train.py --help` can be overriden by `key=value`.

Examples:

 - Use only part of the dataset: `datamodule.include=\"<regex>\"` or `datamodule.exclude=\"<regex>\"`, where `<regex>` is the regular expression for matching the sequences to be included or removed (e.g. `abandonedfactory_Easy_P00[0-2]`)
 - Change the train/validation/test split ratio of dataset: `datamodule.split_ratio=[0.5,0.3,0.2]`.
 - Change the validation interval: `trainer.val_check_interval=<interval>`, see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval).
