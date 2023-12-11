from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter


dir = "/Users/mathisscheffler/Desktop/"
filename = 'huge_pd_shuffled_with_rewards_original_seq4_SHUFFLED.pkl'
savedir = "Desktop/"

# fix random seed for reproducibility
np.random.seed(7)
torch.manual_seed(7)

NUM_LAYERS = 1
HIDDEN_SIZE = 64
TRAIN_EPOCHS = 200

print_iter = 50

TRAIN_NUM_STEPS = 4
TRAIN_BATCHES = 2048
TEST_SIZE = 0

seq_length = 4

with open(dir + filename, 'rb') as f:
    huge_pd = f = pd.read_pickle(f)


def my_train_test_split(X, y, sequence_length,
                        test_size=TEST_SIZE,
                        debug_print=False):

    train_samples = int(np.ceil(X.shape[0] * (1 - test_size)))
    test_samples = int(np.floor(X.shape[0] * test_size))

    if debug_print:
        print("train samples:", train_samples, " test samples:", test_samples)

    divide_train = train_samples % sequence_length == 0

    while not (divide_train):
        train_samples += 1
        divide_train = train_samples % sequence_length == 0

    divide_test = test_samples % sequence_length == 0

    while not (divide_test):
        test_samples -= 1
        divide_test = test_samples % sequence_length == 0

    X_train = X[0:train_samples]
    if debug_print:
        print("X_train indices are 0 to:", train_samples)

    X_test = X[train_samples:].copy()
    if debug_print:
        print("X_test indices are {} to {}:".format(train_samples, len(X)))

    y_train = y[0:train_samples]
    if debug_print:
        print("y_train indices are {} to {}:".format(0, train_samples))

    y_test = y[train_samples:train_samples+len(X_test)]
    if debug_print:
        print("y_test indices are {} to {}:".format(train_samples, len(y)))

    return X_train, X_test, y_train, y_test


cur_data = huge_pd.copy()

cur_data['choice'] = cur_data.choice.apply(lambda x: x - 1)
cur_data['prev_choice'] = cur_data.prev_choice.apply(lambda x: x - 1)  

cur_data = cur_data[cur_data.choice >= 0]
cur_data = cur_data[cur_data.prev_choice >= 0]


X = cur_data.drop(columns=['index', 'choice', 'user', 'time', 'reward',
                           'payoff_structure', 'reward_1', 'reward_2',
                           'reward_3', 'reward_4'])
X_prev = X.prev_choice.to_numpy()
X_prev = F.one_hot(torch.tensor(X_prev, dtype=torch.int64), num_classes=4)

y = cur_data.choice
num_of_classes = len(y.unique())
y = y.to_numpy()
y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.int64),
                                num_classes=4)

new_X = []
for prev_choice, prev_reward in zip(X_prev, X.prev_reward):
    new_i = np.append(prev_choice, prev_reward)
    new_X.append(new_i)
new_X = np.array(new_X)

seq_data = cur_data.reset_index(drop=True).copy()


#  gets a list and return True if the sequence is continuous without gaps,
#  otherwise return False
def is_continous_sequence(choice_numbers):

    for c, choice_number in enumerate(choice_numbers):
        if c == 1:
            continue
        if choice_numbers[c] - choice_numbers[c-1] > 1:
            return False
    return True


start = 0
end = 4

continuous_counter = 0
not_continuous_counter = 0

more_than_one_gap_seq = []
for i in range(start, seq_data.shape[0], 4):
    if i == 0:
        start = 0
        end = start + 4
    start = i
    end = start + 4
    cur_orig_choices = list(seq_data[start:end]['orig_choice_num'])
    if is_continous_sequence(cur_orig_choices):
        continuous_counter += 1
    else:
        not_continuous_counter += 1
        more_than_one_gap_seq.append(cur_orig_choices)


# gets a list and returns : the highest gap in the sequence, total gap
def how_continous_sequence(choice_numbers):
    highest_gap = 0
    gap_sum = 0
    for c, choice_number in enumerate(choice_numbers):
        if c == 1:
            continue
        gap = choice_numbers[c] - choice_numbers[c-1]
        if gap > highest_gap:
            highest_gap = gap
        gap_sum += gap

    return highest_gap, gap_sum


# check the more_than_one_gap_seq - how many gaps there is 
highest_gaps = []
gaps_sums = []
for i in range(len(more_than_one_gap_seq)):
    highest_gap, gap_sum = how_continous_sequence(more_than_one_gap_seq[i])
    highest_gaps.append(highest_gap)
    gaps_sums.append(gap_sum)

highest_gaps_counter = Counter(highest_gaps)
gaps_sums_counter = Counter(gaps_sums)


X_train, X_test, y_train, y_test = train_test_split(new_X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=False)

possible_samples = int(X.shape[0] / seq_length)
possible_samples_train = int(X_train.shape[0] / seq_length)
possible_samples_test = int(X_test.shape[0] / seq_length)
possible_labels_train = int(y_train.shape[0] / seq_length)
possible_labels_test = int(y_test.shape[0] / seq_length)

# reshape X to be [samples, time steps, features]
X_train = np.reshape(X_train[:-1], (possible_samples_train,
                                    seq_length, X_train.shape[1]))

y_cat_train = np.reshape(y_train[:-1], (possible_samples_train,
                                        seq_length, y_train.shape[1]))

X_test = np.reshape(X_test, (possible_samples_test,
                             seq_length, X_test.shape[1]))

y_cat_test = np.reshape(y_test, (possible_labels_test,
                                 seq_length, y_test.shape[1]))

num_of_samples = y_cat_test.shape[0]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


torch.save(X_train, savedir + 'X_train.pt')
torch.save(y_train, savedir + 'Y_train.pt')
torch.save(X_test, savedir + 'X_test.pt')
torch.save(y_test, savedir + 'y_test.pt')
