# Multiway Matching on the Multinomial Manifold

## Publication

* Spyridon Leonardos, Xiaowei Zhou, Kostas Daniilidis, "A Low-Rank Matrix Approximation Approach to Multiway Matching
with Applications in Multi-Sensory Data Association", IEEE International Conference on Robotics and Automation (ICRA), 2020.

## Prerequisites
* numpy
* matplotlib
* scipy

## Run
Sample use:

```bash
python3 test.py --dataset Willow --solver cg --tol 1e-3 --lr 0.05
```
```bash
python3 test.py --dataset synthetic --solver gd --tol 1e-3 --n 10 --k 20 --o 0.3
```
