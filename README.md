# DIRL-bidding_preference

## Backgrounds
The code is used to implement the DIRL algorithm for generators in power markets proposed in the paper: 

H. Guo, Q. Chen, Q. Xia and C. Kang, "Deep Inverse Reinforcement Learning for Reward Function Identification in 
Bidding Models," in IEEE Transactions on Power Systems, doi: 10.1109/TPWRS.2021.3076296.

website: https://ieeexplore.ieee.org/abstract/document/9417741

## Requirements
- NumPy
- SciPy
- CVXOPT
- Theano
- deep_maxent from MatthewJA (details can be found in Acknowledgements)

## Module documentation
The main process of the algorithm can be found in main_DIRL.py, which reads the Markov Decision Process (MDP) data of 
generators from files and uses the DIRL algorithm to reveal the generator's bidding preferences.

The MDP data files (working as inputs) are stored in "/test_data/MDP_information/". Each generator needs three files, whose postfixes are 
"feature", "action" and "state" respectively. The example data files can be found in the project. We have uploaded the 
example data files for BW03, DDPS1, GSTONE4 and STAN-1. If you want to use the code for your data, you have to transform
your data as the example files.

The output results of the code will be stored in "/test_data/reward_function/". The parameters of the ANN used for 
demonstrating the individual bidding preferences will be stored in different ".csv" files. The parameters of the code, reward
features on various states and reward records of each epoch are also stored in different files. The example files can be
found in "/test_data/reward_function/".

The details about variables and parameters in the code are written as annotations. Please directly see the code.

It should be noted that the code is only used for personal academically learning and research exploration. Its efficiency
has not been improved. 

In the near future, our team will publish a more mature version of IRL (DIRL) on market participants with rewriting the codes
of IRL and DIRL algorithms.

If you have any question, feel free to contact me: hyguo@mail.tsinghua.edu.cn

## Acknowledgements
The basic codes for IRL and DIRL used in this project are from Matthew Alger's work, including: maxent.py, deep_maxent.py
, value_iteration.py.

The details about his project can be found on his github website: 
https://github.com/MatthewJA/Inverse-Reinforcement-Learning



