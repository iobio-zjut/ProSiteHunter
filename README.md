# ProSiteHunter
## Install
### Create a Conda Environment
```
conda create -n ProSiteHunter python=3.7.11
conda activate ProSiteHunter
```
### dependencies
Please refer to the requirement.txt file for details on the packages that need to be installed.

## Generate Feature
```
python ./data/utils/process_csv_to_fasta.py
```
Convert CSV to fasta file first

```
python ./data/utils/ProstT5_embedding_generate.py
python ./data/utils/SiteT5_embedding_generate.py

```
Positional encoding, Physicochemical properties and BLOSUM62 will be automatically generated during training or testing. 
Please visit NetSurfP-3.0 online server for RSA and Secondary structure generation (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/)

## Train
```
python ./main/S1131/train/train_S1131.py
```
Run this script to train the S1131 model, the same applies to other datasets.

The weights of the trained model can be downloaded in releases (https://github.com/iobio-zjut/SAMPPI/releases/tag/v1.0)

## Predict
```
python ./main/S1131/predict/predict_S1131.py
```
Run this script to predict the S1131 model, the same applies to other datasets
