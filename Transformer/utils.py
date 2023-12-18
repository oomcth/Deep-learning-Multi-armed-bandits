from torch import save, argmax

from numpy import mean
import matplotlib.pyplot as plt
from tqdm import tqdm


# main training function
def train(model, epochs: int, criterion,
          optimizer, train_loader,
          trainAll=False, echo=False) -> None:

    # list to keep our previous loss
    losses = []

    # setting up our model for training
    # trainAll = True during preTraining False otherwise
    model.trainAll = trainAll
    model.extracting_features = False

    print("training started")

    for epoch in tqdm(range(epochs)):
        # average loss for batch
        running_loss = []
        for batch in train_loader:
            (x, y) = batch

            # forward pass
            outputs, _ = model(x)
            if trainAll:
                loss = criterion(outputs, y)
            else:
                loss = criterion(outputs, y[:, -1, :])

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating loss record
            running_loss.append(loss.item())

        # updating loss record
        losses.append(mean(running_loss))
        running_loss = []

    save(model, "model_TransfoDecoder.pth")

    # plot loss by epoch
    if echo:
        plt.plot(losses)
        plt.show()


# training function for the sparse autoencoder
def extract(model, epochs, optimizer, train_loader, echo=False) -> None:

    # list to keep our previous loss
    losses = []

    # setting up our model for training
    # trainAll = True during preTraining False otherwise
    # we set extracting feature to True as we are extracting features
    model.trainAll = False
    model.extracting_features = True

    print("extraction started")

    for epoch in tqdm(range(epochs)):
        # average loss for batch
        running_loss = []
        for batch in train_loader:

            # forward pass
            (x, _) = batch
            _, loss = model(x)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating loss record
            running_loss.append(loss.item())

        # updating loss record
        losses.append(mean(running_loss))
        running_loss = []

    save(model, "model_TransfoDecoder.pth")

    # plot loss by epoch
    if echo:
        plt.plot(losses)
        plt.show()


# compute the accuracy of our model
def accuray(model, X_train, y_train):

    pred = 0

    for x, y in tqdm(zip(X_train, y_train)):
        temp, _ = model(x.unsqueeze(0))
        if argmax(temp.squeeze()) == argmax(y[-1]):
            pred += 1
    return pred/X_train.size()[0]
