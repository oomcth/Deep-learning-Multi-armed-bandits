# Deep-learning-Multi-armed-bandits

For the LSTM and explicit model experiments, the notebooks should be ran in this order : 

1) preprocess : creates a shuffled csv of choice and rewards from raw choice data

2) general model training and testing with full data : trains the general model with choice data + simulations on synthetic inputs

3) general model training and testing with full data version 2 : trains the general model with choice data + additional feature (reaction time), using MRR metric, and comparison with explicit model

4) no reward model training and testing with full data : trains the reward-blind model on choice data + simulations on synthetic inputs

5) explicit models simulations : simulation of the explicit model on synthetic inputs

6) simulation comparisons : compare simulations of the 3 models, compute and plot the KL divergence

7) figures : figures of accuracy per person to compare models

