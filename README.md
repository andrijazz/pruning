# Pruning Challenge

Analysing pruning strategies ...

### Dependencies

I am using [pipenv](https://pipenv-es.readthedocs.io/es/stable/) for handling python virtual environments:
```
pip3 install --user pipenv
```
and [W&B](https://www.wandb.com/) for visualizations.

### Setup

```
git clone https://github.com/andrijazz/pruning
cd pruning
bash init_env.sh <PROJECT_NAME> <PATH_TO_YOUR_DATA_STORAGE>
pipenv install
```

### References
* https://jacobgil.github.io/deeplearning/pruning-deep-learning
* https://for.ai/blog/targeted-dropout/
* https://arxiv.org/pdf/1905.13678.pdf
