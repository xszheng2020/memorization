# An Empirical Study of Memorization in NLP
> ACL 2022

## Installing / Getting started

```shell
docker run -it --gpus all --name <docker_name> --ipc=host -v <project_path>:/opt/codes nvcr.io/nvidia/pytorch:20.02-py3 bash
pip install torch==1.2.0
pip install transformers==3.0.2
jupyter notebook --notebook-dir=/opt/codes --ip=0.0.0.0 --no-browser --allow-root
```

## Prepare the datasets

Download the **CIFAR-10**, [SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip), [SST](https://nlp.stanford.edu/sentiment/index.html), [Yahoo! Answer](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU) datasets from web and then process them using the 00_EDA.ipynb

## Run the experiments

```shell
git clone https://github.com/xszheng2020/memorization.git
cd cifar
bash ./scripts/run_if_attr_42.sh # compute the memorization scores and memorization attributions
bash ./scripts/run_mem_<X>.sh # train the model while dropping top-X% memorized instances
bash ./scripts/run_random_<X>.sh # train the model while dropping X% instances randomly
bash ./scripts/eval_attr_mem.sh # eval the memorization attributions
bash ./scripts/eval_attr_random.sh # eval the random attributions
```
## Analyze the results
How to analyze the results and plot the most figures in the paper can be found in the jupyter notebooks.

## Links
- Paper: https://arxiv.org/abs/2203.12171
- Related projects:
  - fast-influence-functions: https://github.com/salesforce/fast-influence-functions
