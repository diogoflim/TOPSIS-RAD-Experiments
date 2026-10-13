# TOPSIS-RAD-Experiments

This github repository presents the supplementary python script for the article TOPSIS-RAD: Ranking According to Desires, submited to the Journal of Multi-criteria Decision Analysis, by Leonardo Fernandes Costa, Helder Gomes Costa, Diogo Lima, and Brunno Rodrigues.

The script reproduces the numerical result reported in the paper:
  - Examples A, B, C (Tables 3-16), including the insertion of A11;
  - Example D (Tables 17-19);
  - the simulation study on the frequency of rank reversal (Table 20);
  - the sensitivity analysis of Example D (Table 21).

Requirements: Python 3.9+, numpy, scipy (results in the paper obtained with Python 3.13, numpy 2.5, scipy 1.18).

Usage:  
  - python topsis_rad_experiments.py            (everything; the simulation takes about one minute)
  - python topsis_rad_experiments.py --quick    (skips the simulation and the sensitivity analysis)

All input data are defined in this file; no external file is needed. 

For reproducibility purposes, the random seed is fixed, so the output is identical on every run. 