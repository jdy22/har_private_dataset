import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(device)

logging = True
logfile_name = "LSTM_2p_clean_test4.txt" # CHANGE ME

if logging:
    logfile = open(logfile_name, "w")

# Constants/parameters
window_size = 1000 # Used in pre-processing
batch_size = 10 # Used for training
learning_rate = 0.00001
n_epochs = 2000 # Training epochs
input_dim = 270
hidden_dim = 300
layer_dim = 1
output_dim = 5

if logging:
    logfile.write(f"Window size: {window_size}, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {n_epochs}\n")
    logfile.write(f"Input dimension: {input_dim}, hidden dimension: {hidden_dim}, layer dimension: {layer_dim}, output dimension: {output_dim}\n")
    logfile.write("\n")

# Read in data
print("Reading in data and converting to tensors...")
with open("clean_data_2_fixed_length.pk1", "rb") as file:
    data = pickle.load(file)
x_train = data[0]
x_test = data[1]
y_train = data[2]
y_test = data[3]

# Split training data into train and val data (80/20)
x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.20, random_state=1000)

# Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
x_train_tensor = Variable(torch.Tensor(x_train_train))
x_test_tensor = Variable(torch.Tensor(x_train_val)).to(device=device)
y_train_tensor = Variable(torch.Tensor(y_train_train))
y_test_tensor = Variable(torch.Tensor(y_train_val)).to(device=device)

x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], window_size, -1))
x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], window_size, -1))

# Instantiate LSTM model and loss function
print("Creating LSTM model, loss function and optimiser...")

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Number of hidden units
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True means that input tensor will be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Output layer (linear combination of last outputs)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        h0 = h0.to(device=device)
        c0 = c0.to(device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out.size() --> batch_size, seq_dim, hidden_dim
        # out[:, -1, :] --> batch_size, hidden_dim --> extract outputs from last layer

        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim

        # Apply sigmoid activation to output for multi-label classification
        activation = nn.Sigmoid()
        out = activation(out)

        return out

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
loss_function = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create data loader for batch gradient descent
print("Creating data loader for batches...")

# Dataset builder class
class Dataset_builder(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]        
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index]    
    # Getting length of the data
    def __len__(self):
        return self.len

train_dataset = Dataset_builder(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2)

# Train model
print("Training model...")
model = model.to(device=device)
for n_epoch in range(n_epochs):
    print(f"Starting epoch number {n_epoch+1}")
    for i, (inputs, labels) in enumerate(train_loader):
        # if i%10 == 0:
        #     print(f"{i} batches processed")
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()
        
    with torch.no_grad():
        predictions = model(x_test_tensor)
        labels = y_test_tensor
        test_loss = loss_function(predictions, labels)
        # Calculate accuracy per activity
        accuracies = []
        for i in range(predictions.shape[1]):
            predictions_indiv = predictions[:, i] > 0.5
            labels_indiv = labels[:, i]
            accuracy = torch.count_nonzero(predictions_indiv==labels_indiv)/len(predictions_indiv)
            accuracies.append(accuracy.item())
        average_accuracy = sum(accuracies)/len(accuracies)
        print(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracies}, average accuracy = {average_accuracy}")
        if logging:
            logfile.write(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracies}, average accuracy = {average_accuracy}\n")

cm = multilabel_confusion_matrix(labels.cpu(), (predictions>0.5).cpu())
for i in range(len(cm)):
    print(f"Confusion matrix for activity {i+1}:")
    print(cm[i])

if logging:
    logfile.write(f"\nFinal confusion matrices:\n")
    for i in range(len(cm)):
        logfile.write(f"\nActivity {i+1}:\n")
        for j in range(len(cm[i])):
            logfile.write(str(cm[i, j]) + "\n")
        
    logfile.close()