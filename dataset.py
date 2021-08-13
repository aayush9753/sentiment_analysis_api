import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class DataTraining():  # Dataset Class to load training data and tokenize them and split them in data_train and data_val
    def __init__(self):
        super(DataTraining,self).__init__()
    
    @staticmethod
    def load_data(config):
        data = pd.read_csv(config.data_path)  #  loading the dataset
        if 'Unnamed: 0'in data.columns:  # removing unnecessery columns
            data = data.drop(['Unnamed: 0'], axis=1)
        return data
    
    @staticmethod
    def process_text(text) -> str: 
        process_text = str(text)  #Convert to string
        process_text = process_text.lower()  #Convert string to lower
        process_text = re.sub("<.*?>"," ",process_text)  #Removing html tags
        process_text = re.sub("[^a-zA-Z]"," ",process_text) #Removing all digits and having only letters
        process_text = process_text.split(" ") #Removing all stop words
        process_text = " ".join([word for word in process_text if word not in stopwords.words("english")])
        process_text = " ".join(process_text.split()) # removing extra spaces via spliting and joining
        return process_text
    
    @staticmethod
    def load_tokenizer(tokenizer_path):  # load the tokenizer
        with open(tokenizer_path, 'rb') as handle:  
            tokenizer = pickle.load(handle)
        return tokenizer
    
    @staticmethod
    def create_tokenizer(clean_texts, save_tokenizer_to):  # creating a tokenizer if the user wants from the dataset
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(clean_texts)
        # saving
        with open(save_tokenizer_to, 'wb') as handle:  # saving the tokenizer in user defined path
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer
    
    
    def get_token_vectors(self, tokenizer, list_of_raw_text, max_pad_len):
        text = []
        for i in list_of_raw_text:
            text.append(self.process_text(i))
        sequence_text = tokenizer.texts_to_sequences(text)
        token_vectors = pad_sequences(sequence_text,padding="post", maxlen=max_pad_len)
        return token_vectors
    
    @staticmethod
    def label_to_idx(labels):  # convert label names to label indexes
        label_2_idx = {'negative' : 0, 'positive' : 1}
        idx = np.array([label_2_idx[i] for i in labels])
        return idx
    
    def get_splitted_token_vectors(self, config):
        data = self.load_data(config)
        data_positive = data[data["airline_sentiment"] == "positive"].reset_index(drop = True)
        data_negative = data[data["airline_sentiment"] == "negative"].reset_index(drop = True)
        data = pd.concat([data_positive, data_negative[:len(data_positive)]], axis=0).sample(frac=1, random_state=config.random_state).reset_index(drop = True)
        raw_texts = data["text"].values
        labels = self.label_to_idx(data["airline_sentiment"].values)
        X_train, X_val, y_train, y_val = train_test_split(raw_texts, labels, 
                                                        test_size=config.val_size,
                                                        random_state=config.random_state, shuffle = True, stratify=labels)
        if config.default_tokenizer:
            tokenizer = self.load_tokenizer(config.tokenizer_path)
        else:
            clean_texts = [self.process_text(i) for i in raw_texts]
            tokenizer = self.create_tokenizer(clean_texts, config.save_tokenizer_to)
        
        X_train = self.get_token_vectors(tokenizer, X_train, config.max_pad_len)
        X_val = self.get_token_vectors(tokenizer, X_val, config.max_pad_len)
        
        return X_train, X_val, y_train, y_val