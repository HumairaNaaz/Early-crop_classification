# Environment setup
* Our code: Python 3.10+, `pip install -r requirements.txt`.
* Baseline authors' code (CatCrops_identification, DOI 10.34810/data2322) was built on **Python 3.8.18** (per team plan) -> use a **separate conda env** for reproduction:
  `conda create -n catcrops38 python=3.8 && conda activate catcrops38`
* Compute: Kaggle (about 30 GPU-h/week per account, per plan - verify), Colab free for debugging. Fix seeds; cache tensors; early stopping.
* TODO (Ishu): pin exact versions after the first successful run.
