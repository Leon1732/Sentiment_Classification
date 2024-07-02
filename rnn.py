import pandas as pd
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch, gc

import os
# set max_split_size_mb
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


# download data
tr = pd.read_csv('movie_train.csv', header=None, usecols=[0, 1], skiprows=1)
ts = pd.read_csv('movie_test.csv', header=None, usecols=[0, 1], skiprows=1)

text_tr = tr[0].to_numpy()
label_tr = tr[1].to_numpy()
text_ts = ts[0].to_numpy()
label_ts = ts[1].to_numpy()

enc = tiktoken.get_encoding("cl100k_base")
tokens_tr = []
tokens_ts = []
for int_values in text_tr:
    token = enc.encode(int_values)
    tokens_tr.append(token)
for int_values in text_ts:
    token = enc.encode(int_values)
    tokens_ts.append(token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def truncate_and_pad(arr, target_length, pad_value=0):
    # truncate
    truncated_arr = arr[:target_length]

    padding_length = max(0, target_length - len(truncated_arr))

    # padding
    padded_arr = truncated_arr + [pad_value] * padding_length

    return padded_arr


tokens_tr = tokens_tr
label_tr = label_tr
tokens_ts = tokens_ts
label_ts = label_ts
reshape_tr = []
reshape_ts = []
length = 100
for i in range(len(tokens_tr)):
    reshape = truncate_and_pad(tokens_tr[i], length)
    reshape_tr.append(reshape)
for i in range(len(tokens_ts)):
    reshape = truncate_and_pad(tokens_ts[i], length)
    reshape_ts.append(reshape)

texts_tr = torch.tensor(reshape_tr)
texts_ts = torch.tensor(reshape_ts)
labels_tr = torch.tensor(label_tr)
labels_ts = torch.tensor(label_ts)

texts_tr = texts_tr.to(device)
texts_ts = texts_ts.to(device)
labels_tr = labels_tr.to(device)
labels_ts = labels_ts.to(device)


# define datasets
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# define model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out


# Initialize the model, loss function, and optimizer
vocab_size = 100256  # vocabulary size
embed_dim = 200  # embedding layer dimension
hidden_dim = 256  # hidden layer dimension
output_dim = 1
model = RNNModel(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TextDataset(texts_tr, labels_tr)
loader = DataLoader(dataset, batch_size=256, shuffle=True)
print('start training')

# train the model
epochs = 25
for epoch in range(epochs):
    for text, label in loader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output.squeeze(1), label.float())
        loss.backward()
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

from sklearn.metrics import precision_score, recall_score


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


dataset_ts = TextDataset(texts_ts, labels_ts)
test_loader = DataLoader(dataset_ts, batch_size=128, shuffle=False)

model.eval()
total_correct = 0
total_samples = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        predictions = torch.round(torch.sigmoid(outputs.squeeze(1)))
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = total_correct / total_samples
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
