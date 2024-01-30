# Deep-learning-Multi-armed-bandits

This projects reproduces and explores the methodology presented by Fintz, Osadchy & Hertz to predict and explain human decisions using deep learning 
https://doi.org/10.1038/s41598-022-08863-0


For the LSTM and explicit model experiments, the notebooks should be ran in this order : 

1) preprocess : creates a shuffled csv of choice and rewards from raw choice data

2) general model training and testing with full data : trains the general model with choice data + simulations on synthetic inputs

3) general model training and testing with full data version 2 : trains the general model with choice data + additional feature (reaction time), using MRR metric, and comparison with explicit model

4) no reward model training and testing with full data : trains the reward-blind model on choice data + simulations on synthetic inputs

5) explicit models simulations : simulation of the explicit model on synthetic inputs

6) simulation comparisons : compare simulations of the 3 models, compute and plot the KL divergence

7) figures : figures of accuracy per person to compare models


The transformer related code can be found in the Transformer dossier :

 - model.py is the modified transformer model
 - features_reader.py is a simple tool to visualize features
 - utils.py is a set of usefull methods
 - features_extraction.py trains the sparse autoencoder and the transformer
