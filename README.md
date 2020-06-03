# Pruning Challenge

Analysing pruning strategies.

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

![](https://github.com/andrijazz/pruning/blob/master/docs/plot2.png)

*What do you anticipate the degradation curve of sparsity vs. performance to be? Analyze your results. What interesting insights did you find?*

I was expecting that accuracy will start dropping much sooner. Pruning more then 90% of weights with almost none degradation in performance really surprised me.  

*Do the curves differ? Why do you think that is/isn’t?*
 
Yes. Weight pruning preforms better because pruning the unit discards the portion of information by propagating zeros to the next layer while this is not the case with weight pruning.  

*Do you have any hypotheses as to why we are able to delete so much of the network without hurting performance
(this is an open research question)?*

I like the idea behind lottery ticket hypothesis that certain connections got initial weights such that make training particularly effective - and that those connections form a sub-network capable of preforming the task.

### TODOs
* More experiments - check the results on more complicated networks
* How can we use this for interpretability?
* Implement pytorch SparseLinear module using https://pytorch.org/docs/stable/sparse.html experimental API and measure performance gain
* TF implementation

### References
* https://jacobgil.github.io/deeplearning/pruning-deep-learning
* https://for.ai/blog/targeted-dropout/
* https://arxiv.org/pdf/1905.13678.pdf
