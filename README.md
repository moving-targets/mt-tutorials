# Moving Targets Tutorials

## Requirements

In order to run these tutorials, you will need to install the following packages:

* `cvxpy`
* `gurobipy`
* `jupyter`
* `moving-targets`
* `seaborn`

via the command `pip install -r requirements.txt`.

Moreover, since some optimization problems will rely on the [Gurobi](https://www.gurobi.com/) solver, you will need to 
install it separately. A free academic license for the solver can be obtained
[here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Content

The repository contains different tutorials regarding different known constrained machine learning problems. You can
start a tutorial by opening a terminal and run the command `jupyter notebook`. Once the jupyter browser page will be 
opened, click on the chosen tutorial and learn interactively how to use Moving Targets.

### 00. Introduction

In this tutorial, we will understand the basics of the Moving Targets' algorithm via a toy curve fitting problem. Along
with a theoretical presentation of the method, we will also see some basic functionalities of the library itself, and
how to use it to build fast prototypes leveraging the technique.

### 01. Balanced Counts

In this tutorial we will rely on a wine quality dataset in order to build a custom instance of the algorithm so that
the output classes of this dataset will be balanced. In particular, we will explore the basic objects of the Moving 
Targets library, along with some ways to customize the process in order to have the desired degree of control in all of
your experiments.

### 02. Fair Regression

In this tutorial we will tackle a regression problem instead. Data is retrieved from a cleansed version of the famous
"Communities & Crimes" dataset, thus our goal will be to determine the percentage of violent people in the population of
a certain neighbourhood. Still, since we do not want to penalize marginalized social groups, we will require our model
to satisfy a statistical fairness condition, embodied by the DIDI (Disparate Impact Discrimination Index) metric.

### 03. Fair Classification

In this tutorial we will tackle a similar fairness use case but for a classification task. In particular, data is
retrieved from a cleansed version of the famous "Adult" dataset in which the goal is to determine the income class of a
person ('0' for income >= 50k, '1' otherwise) based on a set of their personal attributes. As well as before, since we
do not want to penalize marginalized social groups, we will use the DIDI metric both to constraint our model and to
evaluate the goodness of our predictions.
