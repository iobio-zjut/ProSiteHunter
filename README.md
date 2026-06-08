# ProSiteHunter

 ProSiteHunter, a unified sequence-based framework for predicting protein binding sites spanning protein-DNA, protein-RNA, protein-protein, and antibody-antigen interfaces. ProSiteHunter integrates the fine-tuned protein language model SiteT5 with evolutionary, geometric, and statistical features extracted from sequences. These representations are further processed through a Multi-Source Feature Fusion (MSFF) module, which captures bidirectional semantics, local associations, and global dependencies to achieve a comprehensive characterization of binding sites, thereby substantially improving predictive accuracy and generalization capability.
 
## Pipeline of ProSiteHunter
![ProSiteHunter pipeline](pipeline.png)
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

The weight files of SiteT5   https://doi.org/10.5281/zenodo.17369404

The generated embeddings of ProstT5 and SiteT5 can be downloaded in the releases. (https://github.com/iobio-zjut/ProSiteHunter/releases/tag/v1.0/)

Positional encoding, Physicochemical properties and BLOSUM62 will be automatically generated during training or testing. 

RSA and secondary structure features can be generated using the NetSurfP-3.0 online server:(https://services.healthtech.dtu.dk/services/NetSurfP-3.0/). 
Alternatively, users may directly download the precomputed feature files from the ProSiteHunter release page:(https://github.com/iobio-zjut/ProSiteHunter/releases/tag/v1.0/)


## Predict
```
python ./main/predict.py
```
