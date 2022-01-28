# Multiway Matching on the Multinomial Manifold

## Overview

* Implementation of conjugate gradient on the multinomial manifold for multiway matching. 
* Support for Willow Motorbike dataset which can be download from [here](https://www.di.ens.fr/willow/research/graphlearning/) 

## Prerequisites
* numpy
* matplotlib
* scipy

## Command line script

```
usage: multiway.py [-h] [--dataset DATASET] [--solver SOLVER] [--i I]
                   [--lr LR] [--tol TOL] [--n N] [--k K] [--o O]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset option: either 'Willow' or 'synthetic' are
                     supported
  --solver SOLVER    Solver option: either 'cgd' for conjugate gradient or
                     'gd' for gradient descent
  --i I              Maximum number of iterations, default to 500
  --lr LR            Learning rate for gradient descent, default to 0.05
  --tol TOL          Convergence parameter, default to 1e-3.
  --n N              Number of images for creating the synthetic dataset
  --k K              Universe size for creating the synthetic dataset
  --o O              Outlier rate for creating the synthetic dataset
```

```bash
python3 multiway.py --dataset Willow --solver cg --tol 1e-3 --lr 0.05
```

```bash
python3 multiway.py --dataset synthetic --solver gd --tol 1e-3 --n 10 --k 20 --o 0.3
```

## Results on the Willow motorbikes dataset

## Citation

* In case you use this code in your research please cite:
  ```
  @inproceedings{leonardos2020low,
    title={A Low-Rank Matrix Approximation Approach to Multiway Matching with Applications in Multi-Sensory Data Association},
    author={Leonardos, Spyridon and Zhou, Xiaowei and Daniilidis, Kostas},
    booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
    pages={8665--8671},
    year={2020},
    organization={IEEE}
  }
  ```
  
  ## Contact 
  
  * Reach out to `spyridon10@gmail.com` for any questions. 
