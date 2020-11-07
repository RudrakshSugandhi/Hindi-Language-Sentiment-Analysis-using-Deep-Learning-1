import pandas as pd
import codecs


df=pd.read_csv("./Intermediate Files/Data.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)


df['text'] = df['text'].apply(lambda x: x.rstrip())
df['text']=df['text'].astype(str)


stopwords=  pd.DataFrame(codecs.open("./Intermediate Files/hindi_stopwords.txt",'r','utf-8'),columns=['words'])
stopwords['words']=stopwords['words'].apply(lambda x: x.rstrip())
stopwords_list=stopwords['words'].tolist()


def remove_stopwords(strn):
    temp=[]
    l=strn.split(" ")
    for word in l:
        flag=0
        for stop in stopwords_list:
            if word==stop:
                flag=1
                break 
        if(flag==0):
            temp.append(word)
    return temp 
df['text']=df['text'].apply(lambda x:remove_stopwords(x))
train_df = df.iloc[:2000,:] 
test_df= df.iloc[2001:,:]

train_df.to_csv("./Intermediate Files/Train.csv")
test_df.to_csv("./Intermediate Files/Test.csv")


