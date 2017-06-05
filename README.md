# Experiments and samples in Multidimensional Scaling
Python script(s) for various MDS algorithms and experiments

### Installation
    git clone https://github.com/mkoledoye/mds_experiments/
    cd mds_experiments
    pip install -r requirements.txt

### Experiments
Run any of the experiments with:

`python run.py --exp_no EXP_NO --nruns NRUNS`

`NRUNS` is the number of times an experiment should be repeated each initialized with random configuration `X`. A good choice is 100, as a higher number allows to have a smooth progression of the plot lines.

`EXP_NO` values are described below.

The experiments are dividing into two groups:

  - Comparisons
        -- compares the performance of MDS variants changing the amount of noise (`EXP_NO=1`) or number of anchors (`EXP_NO=2`).
  - Missing Data
        -- check the effects of missing data in the distance matrix varying the amount of noise (`EXP_NO=3`) or the number of tags (`EXP_NO=4`).

### Animation

A sample animation of the computed configuration using any of the MDS variants can be viewed by running:

`python animation.py`

<sub>NOTE: This requires Python 3</sub>

