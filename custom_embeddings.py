import gensim
import tempfile
from gensim import corpora
from data_processing import train_df,test_df
from gensim.models import FastText

train_sentence=train_df['text'].tolist()
test_sentence=test_df['text'].tolist()

def custom_embeddings(train_list,test_list):
    main_list=train_list+test_list
    model = FastText(main_list,size=50,min_count=1,window=5)
    return model

embedding_model=custom_embeddings(train_sentence,test_sentence)




embedding_model.save("./Intermediate Files/custom_embedding.model")