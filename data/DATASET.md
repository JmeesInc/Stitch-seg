# Dataset Download Instructions

How to obtain Cholec80 and CholecSeg8k used in this experiment.  
Add the instructions below once they are finalized.

---

## Cholec80

please refer to https://camma.unistra.fr/datasets/
or　https://github.com/CAMMA-public/TF-Cholec80.git
```bash
wget https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz
unzip cholec80.tar.gz 
```

---

## CholecSeg8k
```bash
pip install kaggle
kaggle datasets download newslab/cholecseg8k
```