import pandas as pd
from sklearn.utils import shuffle
import codecs
import torch
pos_tweets = codecs.open("./Intermediate Files/pos_train.txt",'r','utf-8')

pos_df=pd.DataFrame(pos_tweets,columns=['text'])
pos_df.insert(1, 'target', '1')
print(len(pos_df))

# negative tweets dataset
neg_tweets =  codecs.open("./Intermediate Files/neg_train.txt",'r','utf-8')
neg_df=pd.DataFrame(neg_tweets,columns=['text'])
neg_df.insert(1, 'target', '0')
print(len(neg_df))

df=pd.concat([pos_df,neg_df],axis=0,ignore_index=True)
df=shuffle(df)

df.to_csv('./Intermediate Files/Data.csv') 



