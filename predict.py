import torch
import numpy as np
from data_processing import remove_stopwords
from Dictionary_Weight_matrix import weights_matrix
from LSTM_NN import LSTM_NN
import pickle 
import CONFIG 


with open("./Intermediate Files/dictionary.pickle", "rb") as input_file:
    dictionary = pickle.load(input_file)

model = LSTM_NN(weights_matrix, CONFIG.output_size, CONFIG.hidden_size, CONFIG.n_layers, drop_prob=0.5)

checkpoint = torch.load('./Intermediate Files/saved_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
def predict(model, dictionary,test_review, sequence_length=27):
    
 model.eval()

# tokenize review
 review_list=remove_stopwords(test_review)

 review_ints = [dictionary[word] for word in review_list]

# pad tokenized sequence
 review_ints.extend([0.0] * abs(sequence_length-len(review_ints)))

 review_ints=np.array(review_ints)

 review_tensor=torch.from_numpy(review_ints)

 review_tensor=review_tensor.unsqueeze(0)

 batch_size = 1

 print(review_tensor.size())

 h = model.init_hidden(batch_size)

 output, h = model(review_tensor, h)

 pred = torch.round(output.squeeze())

 print('Prediction value, pre-rounding: {:.2f}'.format(output.item()))

 if(pred.item()==1):
     print("Positive review detected!")
 else:
     print("Negative review detected.")


predict(model,dictionary,"अच्छी तरह से समाप्त हो गया")
