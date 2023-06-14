# WBC Attribute Prediction

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
