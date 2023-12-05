import math
import random

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import set_random_seed
from tensorflow.python.client import device_lib
import datetime as dt
import os
# from pandas._testing import assert_frame_equal


# fix random seed for reproducibility
np.random.seed(7)
set_random_seed(7)

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
        X_prev = to_categorical(X.prev_choice, dtype='int64')
        y = train.choice
        num_of_classes = len(y.unique())
        y_train = to_categorical(y, dtype='int64')
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
        X_prev = to_categorical(X.prev_choice, dtype='int64')
        y = test.choice
        num_of_classes = len(y.unique())
        y_test = to_categorical(y, dtype='int64')
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
class Model(object):
    def __init__(self,all_x,all_y, is_training, output_size, seq_length, dropout=1.0, batch_size=TRAIN_BATCHES, return_seqence=False):

        # self.x = tf.placeholder(dtype=tf.int32, shape=[None, 4, 5], name='X_placeholder')
        # self.y = tf.placeholder(dtype=tf.int32, shape=[None, 4], name='Y_placeholder')

        # A dataset from a tensor
        dataset = tf.data.Dataset.from_tensor_slices(all_x)
        # Divide the dataset into batches. Once you reach the last batch which won't be 512, the dataset will know exactly which elements remain and should be passed as a batch.
        dataset = dataset.batch(TRAIN_BATCHES)
        # An iterator that can be reinitialized over and over again, therefore having a new shuffle of the data each time
        self.iterator = dataset.make_initializable_iterator()
        # A node that can be run to obtain the next element in the dataset. However, this node will be linked in the model so obtaining the next element will be done automatically
        self.data_X = self.iterator.get_next()

        labels = tf.data.Dataset.from_tensor_slices(all_y)
        # Shuffle the dataset with some arbitrary buffer size
        # dataset = dataset.shuffle(buffer_size=10)
        # Divide the dataset into batches. Once you reach the last batch which won't be 512, the dataset will know exactly which elements remain and should be passed as a batch.
        labels = labels.batch(TRAIN_BATCHES)
        # An iterator that can be reinitialized over and over again, therefore having a new shuffle of the data each time
        self.labels_iterator = labels.make_initializable_iterator()
        # A node that can be run to obtain the next element in the dataset. However, this node will be linked in the model so obtaining the next element will be done automatically
        self.data_Y = self.labels_iterator.get_next()



        self.seq_len = tf.placeholder(dtype=tf.int32,name='sequence_len')


        cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        self.current_batch_size = tf.shape(self.data_X)[0]
        init_state = cell.zero_state(self.current_batch_size, tf.float32)
        self.output, self.states = tf.nn.dynamic_rnn(cell=cell, inputs=tf.cast(self.data_X, tf.float32), initial_state=init_state)

        # tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)

        # reshape to (batch_size * num_steps, HIDDEN_SIZE)
        output = tf.reshape(self.output, [-1, HIDDEN_SIZE])

        softmax_w = tf.Variable(tf.random_uniform([HIDDEN_SIZE, output_size]))
        softmax_b = tf.Variable(tf.random_uniform([output_size]))

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        # if return_seqence:
        self.logits_reshaped = tf.reshape(self.logits, [self.current_batch_size, self.seq_len, output_size])[:,-1,:]

        # TODO: I need return_sequence false , which means I only needs the last output/hidden state
        self.softmax_out = tf.nn.softmax(self.logits)
        self.softmax_out_reshaped = tf.reshape(self.softmax_out, [self.current_batch_size, seq_length, output_size])[:,-1,:]

        loss = tf.keras.losses.categorical_crossentropy(tf.cast(self.data_Y,tf.float32), self.logits_reshaped,from_logits=True)


        self.cost = tf.reduce_sum(loss)

        # TODO: continue
        # get the prediction accuracy
        # self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int64)
        self.predict_return_sequence_false = tf.cast(tf.argmax(self.softmax_out_reshaped, axis=1), tf.int64)

        # self.correct_prediction = tf.equal(tf.argmax(self.predict , 1), tf.argmax(self.data_Y, 1))
        # self.correct_prediction = tf.equal(self.predict, tf.reshape(self.data_Y, [-1]))
        self.correct_prediction_return_sequence_false = tf.equal(self.predict_return_sequence_false, tf.argmax(self.data_Y, 1))

        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy_return_sequence_false = tf.reduce_mean(tf.cast(self.correct_prediction_return_sequence_false, tf.float32))
        # tf.keras.metrics.categorical_accuracy(y_true,y_pred)

        if not is_training:
            return

        # self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.9999).minimize(self.cost)

########################################################################################################
########################################################################################################

def train(all_x, all_y, save_path, seq_length, output_size=4, print_iter=300):
    # setup data and models
    m = Model(all_x,all_y, is_training=True, output_size=4, batch_size=TRAIN_BATCHES, seq_length=seq_length)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([init_op])
        saver = tf.train.Saver()
        accs_over_epochs = []
        loss_over_epochs = []
        counter = 0
        # print("will run for {} steps in total".format(len(all_x)))
        for epoch in range(TRAIN_EPOCHS):
            accs_over_batches = []
            loss_over_batches = []
            sess.run([m.iterator.initializer, m.labels_iterator.initializer])
            try:
                # As long as there are elements execute the block below
                while True:
                    """
                    xs and ys are the current batches
                    output and states are the returned values of tf.nn.dynamic_rnn
                    """
                    xs,ys, \
                    output, states, \
                    logits, logits_reshaped, softmax_out, softmax_out_reshaped,\
                    predict_return_sequence_false, correct_prediction_return_sequence_false,\
                    accuracy_return_sequence_false,\
                        cost, _= sess.run(
                        [m.data_X, m.data_Y,
                         m.output, m.states,
                         m.logits, m.logits_reshaped, m.softmax_out, m.softmax_out_reshaped,
                         m.predict_return_sequence_false, m.correct_prediction_return_sequence_false,
                         m.accuracy_return_sequence_false,
                         m.cost,
                         m.optimizer],
                        feed_dict={
                                   m.seq_len: seq_length
                                   })
                    # print()
                    counter = counter +1
                    accs_over_batches.append(accuracy_return_sequence_false)
                    loss_over_batches.append(cost)
            except tf.errors.OutOfRangeError:
                if epoch%print_iter==0:
                    print("finished epoc {}, acc {} loss {}".format(epoch, np.mean(accs_over_batches),np.mean(loss_over_batches)))
                accs_over_epochs.append(np.mean(accs_over_batches))
                loss_over_epochs.append(np.mean(loss_over_batches))
                # print(accs_over_batches)
                pass
        print("---------------------------------------------------------------")
        print(accs_over_epochs[-1])
        print(np.mean(accs_over_epochs))
        print("loss:")
        print(loss_over_epochs[-1])
        print(np.mean(loss_over_epochs))
        # do a final save
        saver.save(sess, save_path)

        return accs_over_epochs, loss_over_epochs


def test_(all_x, all_y, model_path, seq_length, output_size=4, print_iter=100, print_results=True):
    # setup data and models
    m = Model(all_x,all_y,is_training=False, output_size=4, batch_size=TRAIN_BATCHES, seq_length=seq_length)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([init_op])
        saver.restore(sess, model_path)
        counter = 0
        # print("will run for {} steps in total".format(len(all_x)))
        accs_over_batches = []
        loss_over_batches = []
        predictions_over_batches = []
        correct_predictions = []
        hidden_states = []
        logits_list = []
        softmaxes = []
        outputs = []
        sess.run([m.iterator.initializer, m.labels_iterator.initializer])
        try:
            # As long as there are elements execute the block below
            while True:
                """
                xs and ys are the current batches
                output and states are the returned values of tf.nn.dynamic_rnn
                """
                xs,ys, \
                output, states, \
                logits, logits_reshaped, softmax_out, softmax_out_reshaped,\
                predict_return_sequence_false, correct_prediction_return_sequence_false,\
                accuracy_return_sequence_false,\
                    cost = sess.run(
                    [m.data_X,m.data_Y,
                     m.output, m.states,
                     m.logits, m.logits_reshaped, m.softmax_out, m.softmax_out_reshaped,
                     m.predict_return_sequence_false,  m.correct_prediction_return_sequence_false,
                     m.accuracy_return_sequence_false,
                     m.cost],
                    feed_dict={
                               m.seq_len: seq_length
                               })
                # print()
                counter = counter + 1
                accs_over_batches.append(accuracy_return_sequence_false)
                loss_over_batches.append(cost)
                predictions_over_batches.append(predict_return_sequence_false)
                correct_predictions.append(correct_prediction_return_sequence_false)
                hidden_states.append(states)
                logits_list.append(logits_reshaped)
                softmaxes.append(softmax_out_reshaped)
                outputs.append(output)
        except tf.errors.OutOfRangeError:
            if print_results:
                print("finished testing, acc {} loss {}".format(np.mean(accs_over_batches),np.mean(loss_over_batches)))
                # print(accs_over_batches)
            pass
        if print_results:
            print("---------------------------------------------------------------")
            # print(accs_over_batches[-1])
            print(np.mean(accs_over_batches))
            print("loss:")
            # print(loss_over_batches[-1])
            print(np.mean(loss_over_batches))
        # do a final save
        # saver.save(sess, save_path)

        return accs_over_batches, loss_over_batches, predictions_over_batches, correct_predictions, hidden_states, logits_list, softmaxes, outputs


def train_cv(saving_paths, saving_dirs, fold_X_train, fold_y_train, seq_length):
    # train all folds and save results
    """
    saving_paths : saving dirs for model per sequence
    saving_dirs : saving dirs for results per sequence
    """
    full_train_results = []
    train_results_strings = []
    for path, cur_save_dir, X_train, y_cat_train in zip(saving_paths, saving_dirs, fold_X_train, fold_y_train):
        tf.reset_default_graph()
        accuracies_per_epoch, losses_per_epoch = train(X_train, y_cat_train[:, -1, :], save_path=path, seq_length=seq_length)
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
        tf.reset_default_graph()
        accuracies_test, losses_test, predictions_test, correct_predictions_test, hidden_states, logits_test, softmax_test, outputs_test = test_(X_test, y_cat_test[:, -1, :],
                                                                                                                                                 model_path=path,
                                                                                                                                                 seq_length=seq_length)
        test_results = accuracies_test, losses_test, predictions_test, correct_predictions_test, hidden_states, logits_test, softmax_test, outputs_test
        with open(os.path.join(cur_save_dir, 'test_results.pkl'.format(TRAIN_BATCHES)), 'wb') as handle:
            pickle.dump(test_results, handle)
        print("saved test_results.pkl at {}".format(cur_save_dir))
        full_test_results.append(test_results)
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")
        print(cur_save_dir, "----average test accuracy:", np.average(accuracies_test), "---- highest test accuracy:", np.sort(accuracies_test)[-1])
        results_strings.append(
            "".join([cur_save_dir, "----average test accuracy:", str(np.average(accuracies_test)), "---- highest test accuracy:", str(np.sort(accuracies_test)[-1])]))
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")

    return results_strings, full_test_results


if __name__ == "__main__":
    kfolds = 5

    saved_model_dir = "saved_model/corss_val/general_model/hp_seq_len/"
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    tf.reset_default_graph()

    saving_dir = "cross_validation/general_model/hp_seq_len/"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)


    with open('pd_list_full_with_rewards_original_seq4.pkl', 'rb') as f:
        pd_list = pickle.load(f)

    # use previous seq of 4's to be consistent
    with open('huge_pd_shuffled_with_rewards_original_seq4_SHUFFLED.pkl', 'rb') as f:
        huge_pd_seq_4 = pickle.load(f)

    with open('cross_validation/diff_seq_lengths/all_huge_pd_shuffled_with_rewards_original_seq2_to_seq12_SHUFFLED_LIST.pkl', 'rb') as f:
        huge_pd_hp_list = pickle.load(f)

    # load the kfolded pars
    with open('cross_validation/payoff2_train_participants_5fold_list.pkl', 'rb') as f:
        payoff2_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/payoff2_test_participants_5fold_list.pkl', 'rb') as f:
        payoff2_test_participants_5fold_list = pickle.load(f)

    with open('cross_validation/payoff3_train_participants_5fold_list.pkl', 'rb') as f:
        payoff3_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/payoff3_test_participants_5fold_list.pkl', 'rb') as f:
        payoff3_test_participants_5fold_list = pickle.load(f)

    with open('cross_validation/payoff4_train_participants_5fold_list.pkl', 'rb') as f:
        payoff4_train_participants_5fold_list = pickle.load(f)

    with open('cross_validation/payoff4_test_participants_5fold_list.pkl', 'rb') as f:
        payoff4_test_participants_5fold_list = pickle.load(f)

    seq_lengths_to_evaluate = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # seq_lengths_to_evaluate = [2, 3]
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
            cur_saving_paths.append(os.path.join(save_dir, 'model_with_keras_batch{}'.format(TRAIN_BATCHES)))
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

