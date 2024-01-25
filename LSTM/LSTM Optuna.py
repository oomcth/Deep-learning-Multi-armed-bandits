import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import pickle


# Globals
savedir = ""
Criterion = nn.CrossEntropyLoss()

# Loading Data
with open(savedir + "train_test_valid_data_dict.pkl", 'rb') as f:

    data = pickle.load(f)

    train_, test_, valid_ = data['train'], data['test'], data['valid']
    X_train = list(torch.unbind(train_[0].float()))
    X_test = list(torch.unbind(test_[0].float()))
    X_valid = list(torch.unbind(valid_[0].float()))
    y_train = list(torch.unbind(train_[1].float()))
    y_test = list(torch.unbind(test_[1].float()))
    y_valid = list(torch.unbind(valid_[1].float()))


batch_size = 64
train_loader = DataLoader([[X_train[i], y_train[i]] for i in range(len(X_train))],
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader([[X_test[i], y_test[i]] for i in range(len(X_test))],
                         batch_size=batch_size, shuffle=True)
valid_loader = DataLoader([[X_valid[i], y_valid[i]] for i in range(len(X_valid))],
                          batch_size=batch_size, shuffle=True)


# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size,
                            output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# Training function
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for _ in range(epochs):
        print("+1")
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels[:, :, -1])
            loss.backward()
            optimizer.step()


# Testing function
def test(model, valid_loader, criterion):
    model.eval()
    best_loss = None
    for inputs, labels in valid_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels[:, :, -1])
        Loss = loss.tolist()
        if best_loss is None:
            best_loss = Loss
        elif best_loss > Loss:
            best_loss = Loss
    return best_loss


# Define objective function for Optuna
def objective(trial):

    # Hyperparameter search space
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Model, loss, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = Criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs=epochs)

    # Return the valid loss
    loss = test(model, valid_loader, criterion)
    return loss


# Model fixed parameters
input_size = 5
output_size = 4
epochs = 100

# Train the model with the best hyperparameters
best_model = LSTMModel(input_size, 41,
                       3,
                       output_size)
best_optimizer = optim.Adam(best_model.parameters(),
                            lr=1.4079547470404374e-05)
train(best_model, train_loader, Criterion, best_optimizer, epochs=100)

# save the best_model
try:
    torch.save(best_model.state_dict(), "model.pth")
except:
    torch.save(best_model.state_dict(), "/model.pth")

# return our test error
print(test(best_model, test_loader, Criterion))
input()

# Create optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = LSTMModel(input_size, best_params["hidden_size"],
                       best_params["num_layers"],
                       output_size)

best_optimizer = optim.Adam(best_model.parameters(),
                            lr=best_params["learning_rate"])

train(best_model, train_loader, Criterion, best_optimizer, epochs=10)

# save the best_model
try:
    torch.save(best_model.state_dict(), "model.pth")
except:
    torch.save(best_model.state_dict(), "/model.pth")

# return our test error
print(test(best_model, test_loader, Criterion))
