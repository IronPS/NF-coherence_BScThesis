# Normalizing Flows: A Study On Models' Coherence
TODO

## Disclaimer
This repository is read-only and will receive updates solely with consistency, ease of use or software license compliance in mind.

## Folder Structure
```
.
├── data                        Folder reserved for any data.
│   ├── d01_raw                 Raw data directory.
│   │   └── CelebA_aligned      The CelebA dataset directory.
│   ├── d02_preprocessed        Folder for any processed data. (Not used.)
│   └── d03_model               Folder containining trained models' data.
├── scripts                     Scripts that did not fit anywhere else.
└── src                         Source code directory.
    ├── eval                    Scripts used for evaluating trained models.
    ├── tcc                     Contains all code necessary for training and evaluation.
    └── train                   Training scripts.

```

## Requirements
 - Python 3.5.9
 - The aligned and cropped [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) must be downloaded by hand.
   - The folder structure required is:
        ```
        CelebA_aligned
        ├── images
        │   ├── 000001.png
        │   ├── 000002.png
        │   └── ...
        └── metadata
            ├── list_attr_celeba.txt 
            └── list_eval_partition.txt
        ```

## Installation
```
python3 -mvenv .env
source .env/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

## Pre-trained Models
Both models used during the thesis writing are available [here](https://drive.google.com/drive/folders/1X8leODubgj6uX-47nGyqPhzRgzhkZP8D?usp=sharing). 

Move the downloaded folders to the ```data/d03_model/``` directory.

### Model's Training Folder Structure
Each model's training generates four folders:
```
Model-Execution-Folder
├── Params/             Should have a single JSON file containing the parameters used in training.
├── TestLosses/         Contains the loss calculated with the CelebA test data partition.
├── TrainedFlow/        Models saved through PyTorch (.pt format).
└── TrainLosses/        Contains the loss calculated with the CelebA train data partition.
```

## Known Problems
 - Path separators are hardcoded forward slashes. Windows users may experience problems.
