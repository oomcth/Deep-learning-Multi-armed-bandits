import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna


# Globals
savedir = "Desktop/"
Criterion = nn.CrossEntropyLoss()

# Creating the DataLoaders
X_train = list(torch.unbind(torch.load(savedir + 'X_train.pt')))
y_train = list(torch.unbind(torch.load(savedir + 'Y_train.pt')))
X_test = list(torch.unbind(torch.load(savedir + 'X_test.pt')))
y_test = list(torch.unbind(torch.load(savedir + 'y_test.pt')))
X_valid = list(torch.unbind(torch.load(savedir + 'X_valid.pt')))
y_valid = list(torch.unbind(torch.load(savedir + 'y_valid.pt')))

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
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# Testing function
def test(model, test_loader, criterion):
    model.eval()
    best_loss = None
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
epochs = 10

# Create optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
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
