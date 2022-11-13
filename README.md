# CS4243 Threat Detection

Dataset Google Drive Link: https://drive.google.com/drive/folders/1pCEBqzQDTJ3PlgdIRBY65jOugJ4xpFi6

## Instructions to run the models

### Data requirements

All dataloaders in this repository require a list of strings as the input. The folder structure needs to be /(dataset_folder)/{carrying/normal/threat}. Using pathlib:


``py
def get_data(data_dir):
    data = Path(data_dir).glob('*/*')
    folder_names = ['carrying', 'threat', 'normal']
    data = [x for x in data if x.is_file() and x.suffix != '.zip']
    return data
``

### Saved Model Requirements
Place all saved models in the `models/` directory.

### Running Everything

This repository contains a number of scripts and Jupyter Notebooks. To test the models, run `tester.ipynb'.
