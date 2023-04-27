import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import multilabel_confusion_matrix
from numpy.random import default_rng


def k_fold_split(n_splits, n_instances, random_generator):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """
    # Generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # Split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng(seed=1000)):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """
    # Split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # Pick k as test
        test_indices = split_indices[k]

        # Combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


n_folds = 5 # Total number of folds for cross-validation
fold = 4 # Current fold

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(device)

logging = True
logfile_name = f"ABLSTM_final_fold{fold}.txt" # CHANGE ME

if logging:
    logfile = open(logfile_name, "w")

outfile_name = f"ABLSTM_predictions_fold{fold}.txt"

# Constants/parameters
k = 1  #kernel size for av before lstm
k_2 = 20 # kernel size & stride for av pooling before attention
window_size = int(1000/k) # Used in pre-processing
batch_size = 30 # Used for training
learning_rate = 0.00001
n_epochs = 100# Training epochs
input_dim = 270
hidden_dim = 300
layer_dim = 1
output_dim = 5

if logging:
    logfile.write("Attention Bi-LSTM with Simple Atten & Av. Poolin COMBINED\n")
    logfile.write(f"K size: k:{k} and k_2:{k_2}, Window sizes: before LSTM:{window_size} and before atten:{int(window_size/k_2)}, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {n_epochs}\n")
    logfile.write(f"Input dimension: {input_dim}, hidden dimension: {hidden_dim}, layer dimension: {layer_dim}, output dimension: {output_dim}\n")
    logfile.write("\n")

# Read in data
print("Reading in data and converting to tensors...")
with open("/home/joanna/lstm_model/clean_data_2_fixed_length.pk1", "rb") as file1:
    data1 = pickle.load(file1)
x_train1 = data1[0]
x_test1 = data1[1]
y_train1 = data1[2]
y_test1 = data1[3]

with open("/home/joanna/lstm_model/noisy_data_2_fixed_length.pk1", "rb") as file2:
    data2 = pickle.load(file2)
x_train2 = data2[0]
x_test2 = data2[1]
y_train2 = data2[2]
y_test2 = data2[3]

with open("/home/joanna/lstm_model/clean_data_1_fixed_length.pk1", "rb") as file3:
    data3 = pickle.load(file3)
x_train3 = data3[0]
x_test3 = data3[1]
y_train3 = data3[2]
y_test3 = data3[3]

with open("/home/joanna/lstm_model/noisy_data_1_fixed_length.pk1", "rb") as file4:
    data4 = pickle.load(file4)
x_train4 = data4[0]
x_test4 = data4[1]
y_train4 = data4[2]
y_test4 = data4[3]

x_train = np.vstack((x_train1, x_train2, x_train3, x_train4))
x_test = np.vstack((x_test1, x_test2, x_test3, x_test4))
y_train = np.vstack((y_train1, y_train2, y_train3, y_train4))
y_test = np.vstack((y_test1, y_test2, y_test3, y_test4))

x_full = np.vstack((x_train, x_test))
y_full = np.vstack((y_train, y_test))

# Split data into train and test data (80/20) based on fold number
fold_indices = train_test_k_fold(n_folds, len(x_full))
train_indices, test_indices = fold_indices[fold]

x_train_fold = x_full[train_indices,]
y_train_fold = y_full[train_indices,]
x_test_fold = x_full[test_indices,]
y_test_fold = y_full[test_indices,]

# Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
x_train_tensor = torch.Tensor(x_train_fold)
x_test_tensor = torch.Tensor(x_test_fold).to(device=device)
y_train_tensor = torch.Tensor(y_train_fold)
y_test_tensor = torch.Tensor(y_test_fold).to(device=device)

#incorporate av pooling
avg_pool = nn.AvgPool1d(kernel_size=k, stride=k, count_include_pad=False)

x_train_tensor = x_train_tensor.unsqueeze(1)
x_train_tensor = avg_pool(x_train_tensor)
x_train_tensor = x_train_tensor.squeeze(1)
x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], window_size, -1))

x_test_tensor = x_test_tensor.unsqueeze(1)
x_test_tensor = avg_pool(x_test_tensor)
x_test_tensor = x_test_tensor.squeeze(1)
x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], window_size, -1))

# Instantiate LSTM model and loss function
print("Creating LSTM model, loss function and optimiser...")

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_first=True, bidirectional=True): 
    # def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Number of hidden units
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True means that input tensor will be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True) 
        self.batch_first = batch_first

        # bidirectional
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        # attention layer 
        self.attention = nn.Linear(int(window_size/k_2*(self.D)*hidden_dim), int((window_size/k_2)**2), bias=True,device=device) 

        # Output layer (linear combination of last outputs)
        self.fc = nn.Linear(int(window_size/k_2*(self.D)*hidden_dim), output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim*self.D, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim*self.D, x.size(0), self.hidden_dim).requires_grad_()

        h0 = h0.to(device=device)
        c0 = c0.to(device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        avg_pool_2d = torch.nn.AvgPool2d(kernel_size=(k_2, 1), padding=0, stride=(k_2,1)) 
        out = out.unsqueeze(1)
        out = avg_pool_2d(out)
        out = out.squeeze(1)

        softmax = nn.Softmax(dim=-1)
        relu = nn.ReLU()
        attention = self.attention(out.flatten(start_dim=1,end_dim=-1)) # attention
        attention = softmax(relu(attention.reshape(attention.shape[0],int(window_size/k_2),-1)))
       
        out = torch.bmm(attention, out) #merge
        out = out.flatten(start_dim=1,end_dim=-1) #flatten layer        
        out = self.fc(out)


        # Apply softmax activation to output
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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=5)

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

# Write out predictions of final model
outfile = open(outfile_name, "w")
for i in range(len(predictions)):
    outfile.write(f"True labels: {labels[i, :]}, predicted labels: {(predictions[i, :] > 0.5).long()}\n")
outfile.close()
