# Experiments and samples in Multidimensional Scaling based localization
Python script(s) for various MDS algorithms and experiments

### Installation
    git clone https://github.com/mkoledoye/mds_experiments/
    cd mds_experiments
    pip install -r requirements.txt

### Experiments
Run any of the experiments with:

    python run.py --exp_no EXP_NO --nruns NRUNS

`NRUNS` is the number of times an experiment should be repeated each initialized with a random configuration `X`. Pass `NRUNS >= 100` to have a sufficient amount of trials.

`EXP_NO` values are described below.

The experiments are dividing into two groups:

  - Comparisons:
    compares the performance of MDS variants changing the amount of noise (`EXP_NO=1`) or number of anchors (`EXP_NO=2`).
  - Missing Data:
    check the effects of missing data in the distance matrix varying the amount of noise (`EXP_NO=3`) or the number of tags (`EXP_NO=4`).

### Animation

A sample animation of the computed configuration using any of the MDS variants can be viewed by running:

`python animation.py`

<sub>NOTE: The animation requires Python 3</sub>


### Relevant publications

1. M. A. Koledoye, T. Facchinetti and L. Almeida, "MDS-based localization with known anchor locations and missing tag-to-tag distances," <i>2017 22nd IEEE International Conference on Emerging Technologies and Factory Automation (ETFA)</i>, Limassol, 2017, pp. 1-4.
  [doi: 10.1109/ETFA.2017.8247768][1]
2. C. Di Franco, E. Bini, M. Marinoni and G. C. Buttazzo, "Multidimensional scaling localization with anchors," <i>2017 IEEE International Conference on Autonomous Robot Systems and Competitions (ICARSC)</i>, Coimbra, 2017, pp. 49-54.
  [doi: 10.1109/ICARSC.2017.7964051][2]

[1]: http://ieeexplore.ieee.org/document/8247768/
[2]: http://ieeexplore.ieee.org/document/7964051/
