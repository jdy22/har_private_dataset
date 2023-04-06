import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(device)

logging = True
if logging:
    logfile = open("ABLSTM_1p_clean_test6_simple.txt", "w")

# Constants/parameters
k = 1  #kernel size & stride for av pooling before lstm
k_2 = 10# kernel size & stride for av pooling before attention
window_size = int(1000/k) # Used in pre-processing
batch_size = 50 # Used for training
learning_rate = 0.000005
n_epochs = 50 # Training epochs
input_dim = 270
hidden_dim = 400
layer_dim = 1
output_dim = 5

if logging:
    logfile.write("Attention Bi-LSTM with Av. Poolin 1 person\n")

    logfile.write(f"K size: k:{k} and k_2:{k_2}, Window sizes: before LSTM:{window_size} and before atten:{int(window_size/k_2)}, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {n_epochs}\n")
    logfile.write(f"Input dimension: {input_dim}, hidden dimension: {hidden_dim}, layer dimension: {layer_dim}, output dimension: {output_dim}\n")
    logfile.write("\n")

# Read in data
print("Reading in data and converting to tensors...")
with open("/home/joanna/lstm_model/clean_data_1_fixed_length.pk1", "rb") as file:
    data = pickle.load(file)
x_train = data[0]
x_test = data[1]
y_train = data[2]
y_test = data[3]

# Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
x_train_tensor = Variable(torch.Tensor(x_train))
x_test_tensor = Variable(torch.Tensor(x_test)).to(device=device)
y_train_tensor = Variable(torch.Tensor(y_train))
y_test_tensor = Variable(torch.Tensor(y_test)).to(device=device)

#incorporate av/max pooling

# max_pool = nn.MaxPool1d(kernel_size=k, stride=k)
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
    # def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_first=True, bidirectional=True): 
        super(LSTMModel, self).__init__()
        # Number of hidden units
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True) 
        self.batch_first = batch_first

        # bidirectional
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        # attention layer 
        
        # self.attention = nn.Linear(int(window_size/k_2*(self.D)*hidden_dim), int((window_size/k_2)**2), bias=True,device=device) 
        # self.attention = nn.Linear(int((self.D)*hidden_dim), int(1), bias=True,device=device) 
        # self.attention = nn.MultiheadAttention(int(hidden_dim*self.D),1, bias=True, device=device) 
        self.attention = nn.Linear(int(window_size/k_2*(self.D)*hidden_dim), int((window_size/k_2)**2), bias=True,device=device) 

            # see eqn 3,4,5 in the paper

        # Output layer (linear combination of last outputs)
        self.fc = nn.Linear(int(window_size/k_2*(self.D)*hidden_dim), output_dim)



        # bidirectional can be added thru adding to line 64 init params - "bidirectional=True"
        # LSTM module alrd concat the outputs throughout the seq for us,
        # The outputs of the two directions of the LSTM are concatenated on the last dimension.
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
        # print(out.shape)
        # out.size() --> batch_size, seq_dim, hidden_dim
        # out[:, -1, :] --> batch_size, hidden_dim --> extract outputs from last layer
        
    
        # attention_output, attention_weights = self.attention(out,out,out)
        softmax = nn.Softmax(dim=-1)
        relu = nn.ReLU()
        attention = self.attention(out.flatten(start_dim=1,end_dim=-1)) # attention
        attention = softmax(relu(attention.reshape(attention.shape[0],int(window_size/k_2),-1)))
        # print(attention.shape)
        # print(out.shape)
        # attention = attention.unsqueeze(-1)
        # attention = attention.repeat(1,1,hidden_dim*self.D) # repeat for each hidden dim
        out = torch.bmm(attention, out) #merge
        out = out.flatten(start_dim=1,end_dim=-1) #flatten layer        
        out = self.fc(out)

        # Apply softmax activation to output
        activation = nn.Softmax(dim=1)
        out = activation(out)

        return out

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim) 
loss_function = nn.CrossEntropyLoss()
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
        labels = torch.argmax(labels, dim=1)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()
        
    with torch.no_grad():
        predictions = model(x_test_tensor)
        labels = torch.argmax(y_test_tensor, dim=1)
        # print(predictions.shape)
        # print(labels.shape)
        test_loss = loss_function(predictions, labels)
        accuracy = torch.count_nonzero(torch.argmax(predictions, dim=1)==labels)/len(predictions)
        print(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracy}")
        if logging:
            logfile.write(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracy}\n")

cm = confusion_matrix(labels.cpu(), torch.argmax(predictions, dim=1).cpu(), normalize="true")
print("Confusion matrix:")
print(cm)

output_labels, output_counts = torch.unique(torch.argmax(predictions, dim=1).cpu(), return_counts=True)
print(output_labels)
print(output_counts)

if logging:
    logfile.write(f"\nFinal confusion matrix:\n")
    for i in range(len(cm)):
        logfile.write(str(cm[i]) + "\n")
        
    logfile.close()

