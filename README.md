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
# checkout the code and install all dependencies
git clone https://github.com/andrijazz/pruning
cd pruning
bash init_env.sh <PROJECT_NAME> <PATH_TO_YOUR_DATA_STORAGE>
pipenv install

# activate venv
pipenv shell

# train model
python run.py --mode train

# test models and plot results to w&b
python run.py --mode test

```

### Observations

*What do you anticipate the degradation curve of sparsity vs. performance to be? Your assignment will be to plot this curve for both weight and unit pruning. Compare the curves and make some observations
and hypotheses about the observations.* 

![](https://github.com/andrijazz/pruning/blob/master/docs/plot2.png)

### References
* https://jacobgil.github.io/deeplearning/pruning-deep-learning
* https://for.ai/blog/targeted-dropout/
* https://arxiv.org/pdf/1905.13678.pdf
