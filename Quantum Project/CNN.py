#Import Require Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pickle

import numpy as np
import pandas as pd

import random

from tqdm import tqdm
import time

import matplotlib.pyplot as plt

SEED = 23

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


#Define Custom Dataset
class RNNDataset_OHE(Dataset):

    def __init__(self, fn, length = None):
        #Data Loading
        with open(fn, "rb") as f:
            loaded_data = pickle.load(f)
        all_data = loaded_data[:length]
        self.data = all_data #Return all data as dataframe

        OHE_vecs = np.array(all_data["OHE_matrix"].tolist())
        energies = np.column_stack((all_data["E(S1)"], all_data["E(T1)"]))

        self.OHE = OHE_vecs #matrix of all OHE vectors
        self.energies = energies #Matrix of S1 & T1 energies (X by 2)
        self.n_samples = all_data.shape[0] #number of data points
    
    def __getitem__(self, index):
        #Return Tensors
        ohe_tensor = torch.tensor(self.OHE[index], dtype=torch.float32)
        energies_tensor = torch.tensor(self.energies[index], dtype=torch.float32)
        return ohe_tensor, energies_tensor

    def __len__(self):
        #length of Datatset
        return self.n_samples
    

#Load datasets
path = "./data/OHE.pkl" #Change as required

full_dataset = RNNDataset_OHE(path)

#Splitting dataset 8:1L1
total_size = len(full_dataset)
train_size = int(0.8*total_size)
validation_size = int(0.1*total_size)
test_size = total_size - (train_size + validation_size)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, validation_size, test_size])

#Create DataLoaders for training, validation, & testing
bs = 4

train_dataloader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = bs, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = bs, shuffle = False)


#Convolutional layer with pooling building block
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size, pool_stride):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() #Relu for all layers
        self.pooling = nn.AvgPool2d(pool_size, stride=pool_stride)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x
    
#OHE CNN model
class OHE_SMILES_CNN(nn.Module):
    def __init__(self, args):
        super(OHE_SMILES_CNN, self).__init__()

        #Argument Define
        self.conv_layers_info = args["conv_layers_info"]
        self.fc_in_features = args["conv_layers_info"][-1][0] #Output size of final convolutional layer


        #Create Convolutional Layers using ConvLayer class
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for conv_info in self.conv_layers_info:
            out_channels, kernel_size, stride, padding, pool_size, pool_stride = conv_info
            conv_layer = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, pool_size, pool_stride)
            self.conv_layers.append(conv_layer)
            in_channels = out_channels

        #Create Global Pooling Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)


        #Create Fully Connected Layer
        self.fc = nn.Linear(self.fc_in_features, 2) #2 for the two energies (512,2)

        #Add dropout after the fc layer
        self.dropout = nn.Dropout(p=args["dropout"])


    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  #Flatten the tensor
        x = self.fc(x)
        x = self.dropout(x) #Applies dropout to fc layer

        return x
    
#Define the arguments for the MolecularCNN model
#(out_channels, kernel_size, stride, padding, pool_size, pool_stride) per conv layer
config = {
    "conv_layers_info": [
        (128, 8, 1, 0, 3, 1),
        (256, 4, 1, 0, 4, 2),
        (512, 2, 1, 0, 5, 2)
    ],
    "dropout": 0.1, #Specify dropout rate
    "learning_rate": 0.0005,
    "criterion": nn.SmoothL1Loss()
}

# Create an instance of the MolecularCNN model
model = OHE_SMILES_CNN(config)


learning_rate = config["learning_rate"]
criterion = config["criterion"]
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Traing loop
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    model.train() #Set model to training mode

    r2_scores = 0
    maes = 0
    mses = 0

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        optimizer.zero_grad() #clears gradients calculated from last batch

        #Reshaping input data to match the expected 4D input shape of the CNN
        #Currently input data is [batch_size, embedding_dim, sequence_length] = [4, 35, 160]
        #For CNNs, input tensor is expected to have four dimensions: [batch_size, channels, height, width]
        x = x.unsqueeze(1)
        
        #Forward pass
        y_pred = model(x)

        #Compute the loss
        loss = criterion(y_pred, y)

        #Backpropragation
        loss.backward() #claculate gradient of loss
        optimizer.step() #update parameters by taking an optimizer step

        #print(f"Pred y shape: {y_pred.shape}")
        #print(f"y shape: {y.shape}")

        #Aggregate statistical measures of accuracy
        r2 = r2_score(y.detach().numpy(), y_pred.detach().numpy())
        mae = mean_absolute_error(y.detach().numpy(), y_pred.detach().numpy())
        mse = mean_squared_error(y.detach().numpy(), y_pred.detach().numpy())

        r2_scores += r2
        maes += mae
        mses += mse

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), r2_scores / len(iterator), maes / len(iterator), mses / len(iterator)


#Validation loop (similar to trainig loop but gradients are not calculated)
def evaluate(model, iterator, loss_funct):

    epoch_loss = 0
    model.eval() #Set model to evaluation mode

    r2_scores = 0
    maes = 0
    mses = 0

    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.unsqueeze(1)

            y_pred = model(x)
            loss = loss_funct(y_pred, y)

            #print(f"Pred y shape: {y_pred.shape}")
            #print(f"y shape: {y.shape}")

            #Aggregate statistical measures of accuracy
            r2 = r2_score(y.numpy(), y_pred.numpy())
            mae = mean_absolute_error(y.numpy(), y_pred.numpy())
            mse = mean_squared_error(y.numpy(), y_pred.numpy())

            r2_scores += r2
            maes += mae
            mses += mse

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), r2_scores / len(iterator), maes / len(iterator), mses / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#function to save model when best validation loss is found
def save_model(model, config, filename):
    state = {
        "config": config,
        "model_state_dict": model.state_dict()
    }
    torch.save(state, filename)

#TRAINING LOOP

epochs = 40 #40 in literature

best_valid_loss = float('inf')

train_losses = []
val_losses = []

for epoch in tqdm(range(epochs), desc="Training Model", unit="epoch"):
    start_time = time.monotonic()

    train_loss, train_r2 = train(model, train_dataloader, optimizer, criterion)
    valid_loss, val_r2 = evaluate(model, val_dataloader, criterion)

    train_losses.append(float(train_loss))
    val_losses.append(float(valid_loss))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_model(model, config, "./models/CNN_model.pt")

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if epoch % 5 == 0:
        print(f"Epoch: {epoch} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tTrain R2: {train_r2:.3f}")
        print(f"\tVal. Loss: {valid_loss:.3f}")
        print(f"\tVal. R2: {val_r2:.3f}")

#PLOT LOSSES

plt.figure(figsize=(18, 3)) 

# Plot the first line
plt.plot(train_losses, c="red", label="Training Loss")

# Plot the second line
plt.plot(val_losses, c="blue", label="Validation Loss")

# Add labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation & Training Loss Comparison")
plt.legend()  # Show legend with labels

plt.show()