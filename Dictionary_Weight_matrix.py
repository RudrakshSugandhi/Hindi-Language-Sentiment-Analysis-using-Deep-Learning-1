from gensim.models import FastText
import pickle
import numpy as np
embedding_model=FastText.load("./Intermediate Files/custom_embedding.model")
def build_weight_matrix(model):
    dictionary={}
    matrix_len = len(model.wv.vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    for i, word in enumerate(model.wv.vocab):
        weights_matrix[i] = model[word]
        dictionary[word]=i
    return weights_matrix,dictionary 

weights_matrix,dictionary=build_weight_matrix(embedding_model)    


with open('./Intermediate Files/dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle)