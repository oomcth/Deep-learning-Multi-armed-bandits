import math
import random

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime as dt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from pandas._testing import assert_frame_equal


# fix random seed for reproducibility
np.random.seed(7)

NUM_LAYERS = 1
HIDDEN_SIZE = 64
TRAIN_EPOCHS = 300

print_iter = 50
TRAIN_BATCHES = 2048


def my_kfold_split(huge_pd, kfold_train_payoff2, kfold_train_payoff3, kfold_train_payoff4, kfold_test_payoff2, kfold_test_payoff3, kfold_test_payoff4, k=5):
    """
    gets all the folding (trains and tests pars) and returns (train_data, test_data) in the same amount of folds (if theres 5 folds then 5 train_data etc)
    indices are corresponding ! ([0] in train comes with [0] in test)
    """
    train_data = []
    test_data = []
    for i in range(k):
        train_data.append(huge_pd[(huge_pd['user'].isin(kfold_train_payoff2[i])) |
                                  (huge_pd['user'].isin(kfold_train_payoff3[i])) |
                                  (huge_pd['user'].isin(kfold_train_payoff4[i]))].copy())

        test_data.append(huge_pd[(huge_pd['user'].isin(kfold_test_payoff2[i])) |
                                 (huge_pd['user'].isin(kfold_test_payoff3[i])) |
                                 (huge_pd['user'].isin(kfold_test_payoff4[i]))].copy())
    return train_data, test_data



def get_train_test_data(train_data, test_data, seq_length):
    fold_X_train = []
    fold_y_train = []
    fold_X_test = []
    fold_y_test = []

    for train, test in zip(train_data, test_data):
        X = train.drop(columns=['index', 'choice', 'user', 'time', 'reward', 'payoff_structure', 'reward_1', 'reward_2', 'reward_3', 'reward_4'])
        X_prev = F.one_hot(torch.tensor(np.array(X.prev_choice), dtype = torch.int64), num_classes = 4)
        y = np.array(train.choice)

        y_train = F.one_hot(torch.tensor(y, dtype=torch.int64), num_classes = 4 )
        new_X = []
        for prev_choice, prev_reward in zip(X_prev, X.prev_reward):
            new_i = np.append(prev_choice, prev_reward)
            new_X.append(new_i)
        X_train = np.array(new_X)
        
        possible_samples_train = int(X_train.shape[0] / seq_length)
        possible_labels_train = int(y_train.shape[0] / seq_length)


        # reshape X to be [samples, time steps, features]
        X_train = np.reshape(X_train, (possible_samples_train, seq_length, X_train.shape[1]))
        fold_X_train.append(X_train)
        y_cat_train = np.reshape(y_train, (possible_samples_train, seq_length, y_train.shape[1]))
        fold_y_train.append(y_cat_train)

        ########################################################################
        ### TEST ###############################################

        X = test.drop(columns=['index', 'choice', 'user', 'time', 'reward', 'payoff_structure', 'reward_1', 'reward_2', 'reward_3', 'reward_4'])
        X_prev = F.one_hot(torch.tensor(np.array(X.prev_choice), dtype=torch.int64),  num_classes = 4)
        y = np.array(test.choice)

        y_test = F.one_hot(torch.tensor(y, dtype=torch.int64), num_classes = 4)
        new_X = []
        for prev_choice, prev_reward in zip(X_prev, X.prev_reward):
            new_i = np.append(prev_choice, prev_reward)
            new_X.append(new_i)
        X_test = np.array(new_X)

        possible_labels_test = int(y_test.shape[0] / seq_length)
        possible_samples_test = int(X_test.shape[0] / seq_length)

        X_test = np.reshape(X_test, (possible_samples_test, seq_length, X_test.shape[1]))
        fold_X_test.append(X_test)
        y_cat_test = np.reshape(y_test, (possible_labels_test, seq_length, y_test.shape[1]))
        fold_y_test.append(y_cat_test)

    return fold_X_train, fold_y_train, fold_X_test, fold_y_test


###############################################################################################
###############################################################################################

class Model(nn.Module):
    """
    LSTM model for choice prediction
    """
    def __init__(self, input_size = 5, output_size = 4, batch_size = TRAIN_BATCHES, hidden_size=HIDDEN_SIZE):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True)  # input shape [N, L, H]
        self.linear_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x) # output shape [N, L, H]
        probs = self.linear_layer(output[:, -1, :]) # output shape [N, 4]
        return probs
    
########################################################################################################
########################################################################################################

def train(all_x, all_y, save_path, output_size=4):
    # setup data and models
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(all_x, dtype = torch.float32), torch.tensor(all_y, dtype = torch.float32))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCHES, shuffle=True)
    
    model = Model()
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9999))
    ce_loss = nn.CrossEntropyLoss()
    accs_over_epochs, loss_over_epochs = [],[]

    for epoch in range(TRAIN_EPOCHS):
        accs_over_batches = []
        loss_over_batches = []
        for x_batch, y_batch in train_dataloader :
                
            probs = model(x_batch)     # output of the LSTM

            loss = ce_loss(probs, y_batch.float())   # cross entropy

            _, predictions = torch.max(probs, 1)   # compute accuracy
            _, true_label = torch.max(y_batch, 1)
            accuracy = torch.mean((predictions == true_label).float())

            optimizer.zero_grad()   # gradient descent
            loss.backward()
            optimizer.step()

            accs_over_batches.append(accuracy.item())
            loss_over_batches.append(loss.item())
        
        print("finished epoch {}, acc {} loss {}".format(epoch,np.mean(accs_over_batches),np.mean(loss_over_batches)))
        accs_over_epochs.append(np.mean(accs_over_batches))
        loss_over_epochs.append(np.mean(loss_over_batches))
        del loss
        del accuracy
    print("---------------------------------------------------------------")
    print(accs_over_epochs[-1])
    print(np.mean(accs_over_epochs))
    print("loss:")
    print(loss_over_epochs[-1])
    print(np.mean(loss_over_epochs))

    torch.save(model.state_dict(), save_path)          # Save the model

    return accs_over_epochs, loss_over_epochs


def test_(all_x, all_y, model_path, output_size=4, print_results=True):
    # setup data and models
    
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    ce_loss = nn.CrossEntropyLoss()

    x = torch.tensor(all_x, dtype = torch.float32)
    y = torch.tensor(all_y, dtype = torch.float32)
    probs = model(x)
    
    loss = ce_loss(probs, y.float())   # cross entropy

    _, predictions = torch.max(probs, 1)   # compute accuracy
    _, true_label = torch.max(y, 1)
    accuracy = torch.mean((predictions == true_label).float())
    
    return accuracy.item(), loss.item()


def train_cv(saving_paths, saving_dirs, fold_X_train, fold_y_train, seq_length):
    # train all folds and save results
    """
    saving_paths : saving dirs for model per sequence
    saving_dirs : saving dirs for results per sequence
    """
    full_train_results = []
    train_results_strings = []
    for path, cur_save_dir, X_train, y_cat_train in zip(saving_paths, saving_dirs, fold_X_train, fold_y_train):
        accuracies_per_epoch, losses_per_epoch = train(X_train, y_cat_train[:, -1, :], save_path=path)
        train_results = accuracies_per_epoch, losses_per_epoch
        with open(os.path.join(cur_save_dir, 'train_results.pkl'.format(TRAIN_BATCHES)), 'wb') as handle:
            pickle.dump(train_results, handle)
        full_train_results.append(train_results)
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")
        print(cur_save_dir, "----average train accuracy:", np.average(accuracies_per_epoch), "---- highest train accuracy:", np.sort(accuracies_per_epoch)[-1])
        train_results_strings.append(
            "".join([cur_save_dir, "----average train accuracy:", str(np.average(accuracies_per_epoch)), "---- highest train accuracy:", str(np.sort(accuracies_per_epoch)[-1])]))
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")

    return train_results_strings, full_train_results


# test (evaluation) all folds and save results
def test_cv(saving_paths, saving_dirs, fold_X_test, fold_y_test, seq_length):
    full_test_results = []
    results_strings = []
    for path, cur_save_dir, X_test, y_cat_test in zip(saving_paths, saving_dirs, fold_X_test, fold_y_test):
        accuracy_test, loss_test = test_(X_test, y_cat_test[:, -1, :], model_path=path)
        
        test_results = accuracy_test, loss_test
        with open(os.path.join(cur_save_dir, 'test_results.pkl'.format(TRAIN_BATCHES)), 'wb') as handle:
            pickle.dump(test_results, handle)
        print("saved test_results.pkl at {}".format(cur_save_dir))
        full_test_results.append(test_results)
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")
        print(cur_save_dir, "----average test accuracy:", np.average(accuracy_test))
        results_strings.append(
            "".join([cur_save_dir, "----average test accuracy:", str(np.average(accuracy_test))]))
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")

    return results_strings, full_test_results


if __name__ == "__main__":
    kfolds = 5

    saved_model_dir = "saved_model/cross_val/general_model/hp_seq_len/"
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    saving_dir = "cross_validation/general_model/hp_seq_len/"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    #with open('pd_list_full_with_rewards_original_seq4.pkl', 'rb') as f:
     #   pd_list = pickle.load(f)

    # use previous seq of 4's to be consistent
    #with open('huge_pd_shuffled_with_rewards_original_seq4_SHUFFLED.pkl', 'rb') as f:
    #    huge_pd_seq_4 = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/all_huge_pd_shuffled_with_rewards_original_seq2_to_seq12_SHUFFLED_LIST.pkl', 'rb') as f:
        huge_pd_hp_list = pickle.load(f)

    # load the kfolded pars
    with open('cross_validation/diff_seq_lengths/payoff2_train_participants_5fold_list.pkl', 'rb') as f:
        payoff2_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/payoff2_test_participants_5fold_list.pkl', 'rb') as f:
        payoff2_test_participants_5fold_list = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/payoff3_train_participants_5fold_list.pkl', 'rb') as f:
        payoff3_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/payoff3_test_participants_5fold_list.pkl', 'rb') as f:
        payoff3_test_participants_5fold_list = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/payoff4_train_participants_5fold_list.pkl', 'rb') as f:
        payoff4_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/payoff4_test_participants_5fold_list.pkl', 'rb') as f:
        payoff4_test_participants_5fold_list = pickle.load(f)

    seq_lengths_to_evaluate = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    saving_dirs = []
    for s in seq_lengths_to_evaluate:
        cur_saving_dir = os.path.join(saving_dir, str(s))
        print(cur_saving_dir)
        saving_dirs.append(cur_saving_dir)
        if not os.path.exists(cur_saving_dir):
            os.makedirs(cur_saving_dir)

    saving_models = []
    for s in seq_lengths_to_evaluate:
        cur_saving_dir = os.path.join(saved_model_dir, str(s))
        print(cur_saving_dir)
        saving_models.append(cur_saving_dir)
        if not os.path.exists(cur_saving_dir):
            os.makedirs(cur_saving_dir)

    # generate saving path for each model in cross validation, for each seq_length
    saving_paths_per_seq_for_model = []
    saving_paths_per_seq_for_results = []
    for i, saved_model_per_seq in enumerate(saving_models):
        cur_saving_paths = []
        cur_saving_dirs = []
        for i in range(kfolds):
            save_dir = os.path.join(saved_model_per_seq, 'fold{}/'.format(i))
            cur_saving_dirs.append(save_dir)
            cur_saving_paths.append(os.path.join(save_dir, 'model_with_torch_batch{}'.format(TRAIN_BATCHES)))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        saving_paths_per_seq_for_results.append(cur_saving_dirs)
        saving_paths_per_seq_for_model.append(cur_saving_paths)


    # ACTUAL CV
    for i, s in enumerate(seq_lengths_to_evaluate):
        print("Running seq={}".format(s))
        # if s!=4:
        #     continue
        cur_data = huge_pd_hp_list[i].copy()
        cur_data['choice'] = cur_data.choice.apply(lambda x: x - 1)
        cur_data['prev_choice'] = cur_data.prev_choice.apply(lambda x: x - 1)

        train_data, test_data = my_kfold_split(cur_data,
                                               payoff2_train_participants_5fold_list,
                                               payoff3_train_participants_5fold_list,
                                               payoff4_train_participants_5fold_list,
                                               payoff2_test_participants_5fold_list,
                                               payoff3_test_participants_5fold_list,
                                               payoff4_test_participants_5fold_list,
                                               )
        # sanity checks
        assert (train_data[0].shape != train_data[1].shape)
        assert (train_data[0].shape != train_data[2].shape)
        assert (train_data[0].shape != train_data[3].shape)
        assert (train_data[0].shape != train_data[4].shape)

        assert (train_data[1].shape != train_data[3].shape)
        assert (train_data[1].shape != train_data[4].shape)

        assert (train_data[2].shape != train_data[4].shape)

        fold_X_train, fold_y_train, fold_X_test, fold_y_test = get_train_test_data(train_data, test_data, seq_length=s)

        train_results_strings, full_train_results = train_cv(saving_paths_per_seq_for_model[i], saving_paths_per_seq_for_results[i], fold_X_train, fold_y_train, seq_length=s)

        with open(os.path.join(saving_dirs[i], 'train_results_strings.pkl'), 'wb') as handle:
            pickle.dump(train_results_strings, handle)

        print("Train results for seq={} for all folds:".format(s))
        print(train_results_strings)
        print("###################################")

        test_results_strings, full_test_results = test_cv(saving_paths_per_seq_for_model[i], saving_paths_per_seq_for_results[i], fold_X_test, fold_y_test, seq_length=s)
        print("Test results for seq={} for all folds:".format(s))
        print(test_results_strings)

        with open(os.path.join(saving_dirs[i], 'test_results_strings.pkl'), 'wb') as handle:
            pickle.dump(test_results_strings, handle)

    print("----------------------------------------------------------")


