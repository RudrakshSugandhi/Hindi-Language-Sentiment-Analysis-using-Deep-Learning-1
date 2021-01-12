import streamlit as st 
import CONFIG 
import torch
from pad_sequence import pad_features
from build_vocab import vocab_to_int
from LSTM_NN import LSTM_NN
import numpy as np
import pickle
st.title("Hindi Sentiment Analysis")
sequence_length=27
with open("./Intermediate Files/vocab_dictionary.pickle", "rb") as input_file:
    dictionary = pickle.load(input_file)

model = LSTM_NN(CONFIG.vocab_size, CONFIG.output_size,CONFIG.embedding_size,CONFIG.hidden_size, CONFIG.n_layers, drop_prob=0.5)

checkpoint = torch.load('./Intermediate Files/saved_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

def tokenize_review(test_senti):
    test_words = test_senti.split()
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

model.eval()
sequence_length=47

sentence = st.text_input('Input your sentence here:')

b=st.button('Predict Sentiment')

if b:

    review_ints = tokenize_review(sentence)

    review_ints=pad_features(review_ints,sequence_length)

    review_tensor=torch.from_numpy(review_ints)

    batch_size = review_tensor.size(0)

    h = model.init_hidden(batch_size)

    output, h = model(review_tensor, h)

    pred = torch.round(output.squeeze())

    pred_float=float(pred)

    if(pred_float>=0.5):
        st.write("Positive review detected!")
    else:
        st.write("Negative review detected.")

    st.write('Confidence is:',pred_float)