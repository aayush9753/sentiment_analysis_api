from tqdm.auto import tqdm
import pickle

import re
from nltk.corpus import stopwords
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import Network
from test_configs import config

class Inference(torch.nn.Module):
    def __init__(self):
        super(Inference, self).__init__()

    @staticmethod
    def load_tokenizer(tokenizer_path):  # load the tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    @staticmethod
    def idx_to_label(idx):  # convert label indexes to label names
        idx_2_label = {0: 'negative', 1: 'positive'}
        label = idx_2_label[idx]
        return label

    @staticmethod
    def process_text(text_string) -> str:
        process_text = str(text_string)  # Convert to string
        process_text = process_text.lower()  # Convert string to lower
        process_text = re.sub("<.*?>", " ", process_text)  # Removing html tags
        process_text = re.sub("[^a-zA-Z]", " ", process_text)  # Removing all digits and having only letters
        process_text = process_text.split(" ")  # Removing all stop words
        process_text = " ".join([word for word in process_text if word not in stopwords.words("english")])
        process_text = " ".join(process_text.split())  # removing extra spaces via splitting and joining
        return process_text

    def model_output_for_single_text(self, text_string, tokenizer, max_pad_len, model):  # model output for a single text
        processed_text = self.process_text(text_string)  # processing of the text
        sequence_text = tokenizer.texts_to_sequences(processed_text)  # tokenization of the text
        t = []  # sequence_text consists in the form of [[1], [2], [3]]
        for i in sequence_text: 
            if len(i) != 0:
                t.append(i[0])
        t = [t] # converting in the form of [[1, 2, 3]]
        sequence_text = pad_sequences(t, padding="post", maxlen=max_pad_len)  # padding of the tokens
        tensor = torch.tensor(sequence_text)  # taking Transpose to get the Tensor in desirable format
        y_pred = torch.sigmoid(model(tensor)).item()  # Predicting model output
        pred_idx = round(y_pred)  # rounding
        pred_label = self.idx_to_label(pred_idx)  # label index ==> label name
        prediction = {"label" : pred_label,
                      "negative" : 1 - y_pred,
                      "positive" : y_pred}
        return prediction 

    def predict(self, text, config=config):

        tokenizer = self.load_tokenizer(config.tokenizer_path)  # load tokenizer
        vocab_size = len(tokenizer.word_index) + 1  # vocab length of the tokenizer
        model = Network(in_neuron=vocab_size, num_layers=config.num_layers,
                        embedding_dim=config.embedding_dim, hidden_size=config.hidden_size,
                        out_neuron=config.out_neuron, drop=config.drop)  # initialisation of model
        model.load_state_dict(torch.load(config.path_2_saved_model))  # loading pretrained weights
        model.eval()  # model ==> eval mode
        # checking if text is a single datapoint or list of data points
        # output for single text
        return self.model_output_for_single_text(text, tokenizer, config.max_pad_len, model)

def predict(text, conifiguration = config):
    inf = Inference()
    if type(text) == str:
        prediction = inf.predict(text, conifiguration)
        return prediction
    else:
        predictions= []
        for i in tqdm(text):
            predictions.append(inf.predict(text, conifiguration))
        return predictions
