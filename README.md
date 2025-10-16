# ProSiteHunter
## Install
### Create a Conda Environment
```
conda create -n ProSiteHunter python=3.7.11
conda activate ProSiteHunter
```
### ProSiteHunter dependencies
Please refer to the requirement.txt file for details on the packages that need to be installed.

### ProstT5 and SiteT5 dependencies
```
pip install torch
pip install transformers
pip install sentencepiece
```
## Generate Feature
```
python ./ProstT5_embedding_generate.py
python ./SiteT5_embedding_generate.py
```
ProstT5  https://github.com/mheinzinger/ProstT5

The generated embeddings of ProstT5 and SiteT5 can be downloaded in the releases.（https://github.com/iobio-zjut/ProSiteHunter/releases/tag/v1.0）

Positional encoding, Physicochemical properties and BLOSUM62 will be automatically generated during training or testing. 

Please visit NetSurfP-3.0 online server for RSA and Secondary structure generation (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/)

## Train and Predict
```
python ./main/train.py
python ./main/predict.py
```
