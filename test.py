import torch
import pickle
import CONFIG
from LSTM_NN import LSTM_NN

import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from Dictionary_Weight_matrix import weights_matrix
from data_processing import test_df

with open("./Intermediate Files/dictionary.pickle", "rb") as input_file:
    dictionary = pickle.load(input_file)

model = LSTM_NN(weights_matrix, CONFIG.output_size, CONFIG.hidden_size, CONFIG.n_layers, drop_prob=0.5)
optimizer = torch.optim.Adam( model.parameters(), lr=CONFIG.learning_rate)
criterion=nn.BCELoss()
checkpoint = torch.load('./Intermediate Files/saved_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_sentence=test_df['text'].tolist()
test_label=test_df['target'].values

def sentence_to_int(li):
    f_list=[]
    for sent in li:
        new_list=[]
        for word in sent:
            new_list.append(dictionary[word])
        t=torch.LongTensor(new_list)
        f_list.append(t)    
    return f_list

test_sentence_int=sentence_to_int(test_sentence)

test_tensor=pad_sequence(sequences=test_sentence_int,batch_first=True,padding_value=0.0)

test_data = TensorDataset(test_tensor, torch.from_numpy(test_label))

test_loader = DataLoader(test_data, shuffle=True, batch_size=CONFIG.batch_size,drop_last=True)


test_losses = []
num_correct = 0
h = model.init_hidden(CONFIG.batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))