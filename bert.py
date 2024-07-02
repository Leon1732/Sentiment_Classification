import pandas as pd
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch, gc
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

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

# import bert
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# define datasets
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = TextDataset(text_tr, label_tr, tokenizer)
test_dataset = TextDataset(text_ts, label_ts, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# train the model
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
predictions = []
labels = []

# test the model
with torch.no_grad():
    for batch in test_loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions.append(logits.argmax(dim=-1).numpy())
        labels.append(batch['labels'].numpy())