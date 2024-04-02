# WBC Attribute Prediction

- April 2, 2024. Added pretrained models to replicate the reported results +  some minor updates.

## Environment
We use pytorch version 2.0.0 with python 3.8.16. See `requirements.txt` for details.

## Data Preparation
### PBC dataset.
- Download the data from here: [PBC Dataset](https://data.mendeley.com/datasets/snkd93bnjr/1)
- Unzip it and make a `PBC_dataset_normal_DIB` folder under `./data/PBC` or anywhere of your choice.

### Our dataset (included here)
- `pbc_attr_v1_train.csv`, `pbc_attr_v1_val.csv`, and `pbc_attr_v1_test.csv` contain attribute annotations for the train/val/test splits.
    - `cell_size, cell_shape, nucleus_shape, nuclear_cytoplasmic_ratio, chromatin_density, cytoplasm_vacuole, cytoplasm_texture, cytoplasm_colour, granule_type, granule_colour, granularity`: The attribute columns.
    - `img_name`: This is the image file name. It can serve as a unique identifier.
    - `label`: One of the five WBC types (neutrophils, eosinophils, basophils, monocytes, and lymphocytes) provided by the PBC dataset.
    - `path`: Image path organized by the PBC dataset.

## How to run
```
python traineval.py  --help
python traineval.py  --image_dir ./data/PBC # make sure to adjust the place of the PBC dataset.
```
Note: Please make sure to adjust the args and any other details based on your specific setup.

## Pretrained Model
- https://www.dropbox.com/scl/fi/clewulqq4qkuk9cusrttm/resnet50_6c33f0.pth?rlkey=wp2qf0z2jt95npnq46d9xy61k&dl=0
    - md5hash: `6c33f0659662efffb3ed5f91d63abbee`
    - The model is trained by `python traineval.py --seed 2 --use_eval_mode`
- `python traineval.py --resume resnet50_6c33f0.pth --epochs 0`
	- This should give the results of `acc: 94.09, f1_macro: 91.27, pre_macro: 90.86, rec_macro: 91.77`