#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import csv
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

import copy
from PIL import Image
from itertools import cycle


# for importing data
import torchvision
import os
import sys
print(sys.executable)

# for getting summary info on models
#from torchsummary import summary

# for data visualization
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')


# In[2]:


file_path_a = "/lustre/work/bagheri/sazizian/PhD/data_12_26_2023/ABCF1/ABCF1.107.fa"
file_path_b = "/lustre/work/bagheri/sazizian/PhD/data_12_26_2023/ABCF1/ABCF1_HUMAN.pssm"
file_path_c = "/lustre/work/bagheri/sazizian/PhD/data_12_26_2023/ABCF1/ResPRE.pro"


# Function to process sequences
def process_sequences(sequences, letter2number):
    processed_seqs = torch.zeros((len(sequences), len(sequences.iloc[0])), dtype=torch.long)
    for i, seq in enumerate(sequences):
        processed_seq = [letter2number.get(ch, 0) for ch in seq]
        processed_seqs[i] = torch.tensor(processed_seq)
    return processed_seqs


# Function to load PSSM data from a file
def load_pssm(file_path):
    """
    Loads a PSSM file and returns the PSSM matrix.
    """
    alphabet = 'ARNDCEQGHILKMFPSTWYV'  # Standard 20 amino acids
    pssm = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 44:  # Lines with PSSM data have 44 parts
                scores = [int(x) for x in parts[2:22]]
                pssm.append(scores)
    return np.array(pssm), alphabet
    
def process_pssm(pssm, alphabet):
    """
    Convert the PSSM to an expanded one-hot encoding with shape (sequence_length, 20, 20).
    Each position in the sequence will be represented by a 20x20 matrix.
    """
    # Initialize the expanded one-hot matrix
    expanded_one_hot = np.zeros((pssm.shape[0], len(alphabet), len(alphabet)))

    # For each position in the PSSM, create a 20x20 matrix
    for i in range(pssm.shape[0]):
        for j in range(len(alphabet)):
            expanded_one_hot[i, j, j] = 1  # Set the diagonal to 1

    return expanded_one_hot

# Function to process Contact Map data
def process_contact_map(file_path):
    protein_structre = pd.read_csv(file_path, sep='\s+')
    # Create a 750x750 matrix filled with '*'
    protein_matrix = np.full((750, 750), 0)
    for index, row in protein_structre.iterrows():
        i = int(row['i'])
        j = int(row['j'])
        p = float(row['p'])
    
    # Check the condition and assign the value
    if p > 0.50:
        protein_matrix[i][j] = 1  # subtract 1 because Python uses 0-based indexing
        protein_matrix[j][i] = 1


    # Convert the numpy array to a pandas DataFrame for better visualization
    df_protein_matrix = pd.DataFrame(protein_matrix)
    return df_protein_matrix

# Define a separate class for the main file (with target)


# In[3]:


# Load RNA data (Replace with your data loading code)

data_a = pd.read_csv(file_path_a, sep='\t', names=['sequence', 'label'])
data_labels_a = data_a['label']
data_sequences_a = data_a['sequence']

# Process RNA Sequences
unique_characters_a = set(''.join(data_sequences_a))
number2letter_a = dict(enumerate(unique_characters_a))
letter2number_a = {l: i for i, l in number2letter_a.items()}
processed_sequences_a = process_sequences(data_sequences_a, letter2number_a)
print(f"Shape of processed_sequences_a: {processed_sequences_a.shape}")


# Load PSSM data (Replace with your data loading code)
pssm, alphabet =load_pssm(file_path_b)
processed_pssm_b = pssm
#processed_pssm_b = process_pssm(pssm, alphabet)
# Print the shape after processing
print(f"Shape of processed PSSM data: {processed_pssm_b.shape}")


# Load Contact Map data (Replace with your data loading code)

processed_protein_matrix = process_contact_map(file_path_c)  # Replace with your actual path
# Ensure that the processed_protein_matrix has the correct shape (750x750)
processed_protein_matrix = processed_protein_matrix.iloc[:750, :750].values
# Now processed_protein_matrix has the shape (750, 750)
print(f"Shape of processed_protein_matrix: {processed_protein_matrix.shape}")

# Repeat or tile PSSM and contact map data to match the number of RNA sequences
num_rna_sequences = processed_sequences_a.shape[0]
repeated_pssm = np.tile(processed_pssm_b, (num_rna_sequences // processed_pssm_b.shape[0] + 1, 1, 1))[:num_rna_sequences]
repeated_contact_map = np.tile(processed_protein_matrix, (num_rna_sequences // processed_protein_matrix.shape[0] + 1, 1, 1))[:num_rna_sequences]

print(repeated_pssm.shape)
print(repeated_contact_map.shape)


# # Step 1: Define Sub-Models

# In[4]:


class RNASubModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNASubModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last LSTM cell
        x = self.fc(x)
        return x

class PSSMSubModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PSSMSubModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last LSTM cell
        x = self.fc(x)
        return x


class ContactMapSubModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ContactMapSubModel, self).__init__()
        # Assuming the contact map is a 2D matrix
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim * input_dim * hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))  # Add channel dimension
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, rna_model, pssm_model, contact_model, combined_feature_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.rna_model = rna_model
        self.pssm_model = pssm_model
        self.contact_model = contact_model
        self.fc = nn.Linear(combined_feature_dim, 1)
        
    def forward(self, rna_data, pssm_data, contact_data):
            print("Entering forward pass")
            
            # Temporarily bypass parts of the model to isolate the issue
            try:
                rna_features = self.rna_model(rna_data)
                #print(f"RNA features: {rna_features}")
                pssm_features = self.pssm_model(pssm_data)
               # print(f"PSSM features: {pssm_features}")
                contact_features = self.contact_model(contact_data)
               # print(f"Contact features: {contact_features}")
        
                combined_features = torch.cat((rna_features, pssm_features, contact_features), dim=1)
               # print(f"Combined features: {combined_features}")
        
                x = self.fc(combined_features)
               # print(f"Output: {x}")
    
                return x
            except Exception as e:
                print(f"Error in forward pass: {e}")
                raise e



# # Step 2: Define the Combined Model
# The Combined Model will take the outputs from each of the sub-models, concatenate them, and then apply one or more fully connected layers for the final prediction. Here's an example implementation
# ters::

# In[5]:


class ContactMapSubModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContactMapSubModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Calculate reduced dimension after three pooling layers
        reduced_dim = input_dim // 2 // 2 // 2  # Assuming each pooling layer halves each dimension
        self.fc = nn.Linear(128 * reduced_dim * reduced_dim, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# In[6]:


# Example instantiation
vocab_size = 20  # Adjust based on your RNA data
rna_embedding_dim = 64
rna_hidden_dim = 128
rna_output_dim = 50

pssm_input_dim = 20  # Adjust based on your PSSM data
pssm_hidden_dim = 128
pssm_output_dim = 50

contact_input_dim = 750  # Adjust based on your contact map data
contact_output_dim = 50

combined_feature_dim = rna_output_dim + pssm_output_dim + contact_output_dim
output_dim = 2  # Adjust based on your prediction task

rna_model = RNASubModel(vocab_size, rna_embedding_dim, rna_hidden_dim, rna_output_dim)
pssm_model = PSSMSubModel(pssm_input_dim, pssm_hidden_dim, pssm_output_dim)
contact_model = ContactMapSubModel(contact_input_dim, contact_output_dim)  # Corrected

combined_model = CombinedModel(rna_model, pssm_model, contact_model, combined_feature_dim, output_dim)

# Print the model to verify its structure



# In this model:
# 
# rna_model, pssm_model, and contact_model are instances of the respective sub-models.
# combined_feature_dim is the total size of the concatenated features from all three sub-models. It's the sum of the output dimensions of each sub-model.
# output_dim is the size of the final output (e.g., number of classes in classification).
# You'll need to instantiate this model with the sub-models and their parameters:

# # Step 3: Prepare Data Loaders
# We'll create data loaders for RNA sequences. For PSSM and contact map data, given that they are used in full for each batch of RNA sequences, we'll handle them differently.
# 
# Preparing Data Loaders for RNA Sequences
# Assuming you have already processed your RNA sequence data (processed_sequences_a) and labels (data_labels_a), you'll split this data into training and testing sets and create data loaders from them.
# 
# Here is an example of how to create these data loaders:

# In[7]:


from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
train_seqs, test_seqs, train_labels, test_labels = train_test_split(
    processed_sequences_a, data_labels_a, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_seqs_tensor = torch.tensor(train_seqs, dtype=torch.long)
test_seqs_tensor = torch.tensor(test_seqs, dtype=torch.long)

# Convert pandas Series to numpy array before converting to PyTorch tensors
train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(train_seqs_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_seqs_tensor, test_labels_tensor)

# Create DataLoaders
batch_size = 128  # You can adjust this value
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Handling PSSM and Contact Map Data
# Since the entire PSSM and contact map datasets are used for each batch, you will pass these datasets directly to the model in your training and testing loops, rather than using a DataLoader. Ensure that these datasets are in a tensor format compatible with your model:

# In[8]:


# Assuming processed_pssm_b and processed_protein_matrix are already numpy arrays
# Convert PSSM and contact map data to PyTorch tensors
pssm_tensor = torch.tensor(processed_pssm_b, dtype=torch.float32)
contact_map_tensor = torch.tensor(processed_protein_matrix, dtype=torch.float32)


# # Step 4: Training and Validation
# 
# We can encapsulate the training and validation process within a method. This makes the code more organized and reusable. We'll define a method called train_and_validate that takes the model, data loaders, loss function, optimizer, and the number of epochs as inputs and performs both training and validation.
# 
# Here's the updated code with the train_and_validate method:

# In[72]:


def train_step_test(model, train_loader, pssm_tensor, contact_map_tensor, loss_function, optimizer):
    model.train()  # Set the model to training mode

    sequences, labels = next(iter(train_loader))
    batch_pssm = pssm_tensor.repeat(sequences.size(0), 1, 1)
    batch_contact_map = contact_map_tensor.repeat(sequences.size(0), 1, 1, 1)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(sequences, batch_pssm, batch_contact_map)
    # Convert labels to float and add dimension
    labels = labels.unsqueeze(1).type_as(outputs)  

    loss = loss_function(outputs, labels)

    # Backward pass
    loss.backward()
    # After loss.backward(), add:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


    # Optimization step
    optimizer.step()
    print("Training step completed. Loss:", loss.item())

# Define the loss function for binary classification
loss_function = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.01)



# Call the training step test function
train_step_test(combined_model, train_loader, pssm_tensor, contact_map_tensor, loss_function, optimizer)


# In[ ]:


from torch.utils.data import DataLoader
import torch

# Assuming the combined_model, pssm_tensor, and contact_map_tensor are already defined
# Define the loss function for binary classification
loss_function = nn.BCEWithLogitsLoss()

# Define the optimizer using SGD with a lower learning rate
optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.001)

# Training and Validation Function
def train_and_validate(model, train_loader, test_loader, pssm_tensor, contact_map_tensor, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (sequences, labels) in enumerate(train_loader):
            print(f"Processing batch {i+1}/{len(train_loader)}")
            batch_pssm = pssm_tensor.repeat(sequences.size(0), 1, 1)
            batch_contact_map = contact_map_tensor.repeat(sequences.size(0), 1, 1, 1)

            optimizer.zero_grad()
            outputs = model(sequences, batch_pssm, batch_contact_map)
            labels = labels.unsqueeze(1).type_as(outputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
            
        print(f"Epoch {epoch+1} training complete. Total Loss: {total_train_loss}")

        # Add validation code here
        # Validation
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in test_loader:
                batch_pssm = pssm_tensor.repeat(sequences.size(0), 1, 1)
                batch_contact_map = contact_map_tensor.repeat(sequences.size(0), 1, 1, 1)

                outputs = model(sequences, batch_pssm, batch_contact_map)
                labels = labels.unsqueeze(1).type_as(outputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(f"predicted [{predicted}]")
                total += labels.size(0)
                correct += (predicted == labels.squeeze(1)).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_train_loss/len(train_loader)}, Accuracy: {accuracy}%")

# Training and validation
num_epochs = 30  # Adjust the number of epochs as needed
train_and_validate(combined_model, train_loader, test_loader, pssm_tensor, contact_map_tensor, loss_function, optimizer, num_epochs)


# In[ ]:





# In[ ]:




