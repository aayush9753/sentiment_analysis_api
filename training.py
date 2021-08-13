import numpy as np
from tqdm.auto import tqdm
import pickle
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from model import Network


class Dataset(Dataset):  # Default dataset creation function for PyTorch
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X = torch.tensor(self.X[index])
        Y = torch.tensor(self.Y[index])
        return [X, Y]

    def __len__(self):
        return len(self.Y)

class Training(torch.nn.Module):
    def __init__(self):
        super(Training, self).__init__()
    
    @staticmethod
    def get_dataloaders(X_train, X_val, y_train, y_val, batch_size): # creating dataloader
        train_loader = DataLoader(Dataset(X_train, y_train), batch_size = batch_size, shuffle=True)
        val_loader = DataLoader(Dataset(X_val, y_val), batch_size = len(y_val), shuffle=True)
        return train_loader, val_loader

    @staticmethod
    def load_tokenizer(tokenizer_path):  # loading tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    
    @staticmethod
    def idx_to_label(idxs):  # convert label indexes to label names
        idx_2_label = {0 : 'negative', 1 : 'positive'}
        labels = np.array([idx_2_label[i] for i in idxs])
        return labels
    
    @staticmethod
    def train_one_epoch(network, train_iter, optimizer, loss_fn, epoch_num):  # train the model for 1 epoch

        epoch_loss = 0 # loss per epoch
        epoch_acc = 0 # accuracy per epoch

        network.train() # set the model in training mode as it requires gradients calculation and updtion
        # turn off while testing using  model.eval() and torch.no_grad() block

        batch_count = 0
        for batch in tqdm(train_iter,f"Epoch: {epoch_num}"): 
            # data will be shown to model in batches per epoch to calculate gradients per batch

            X = batch[0]
            Y = batch[1].to(torch.float32)
            optimizer.zero_grad() # clear all the calculated grdients from previous step

            predictions = network(X)

            loss = loss_fn(predictions, Y.unsqueeze(1)) # calculate loss on the whole batch

            pred_classes = torch.round(torch.sigmoid(predictions))[:, 0]
            correct_preds = (pred_classes == Y).float()
            # get a floating tensors of predicted classes  which match original true class 

            accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]

            # below two are must and should be used only after calculation of Loss by optimizer
            loss.backward() # Start Back Propagation so that model can calculate gradients based on loss
            optimizer.step() # update the weights based on gradient corresponding to each neuron

            epoch_loss += loss.item()  # add the loss for this batch to calculate the loss for whole epoch
            epoch_acc += accuracy.item() # .item() tend to give the exact number from the tensor of shape [1,]


            time.sleep(0.001) # for tqdm progess bar
            if batch_count % 1 == 0:
                print(f"Batch : {batch_count}/{len(train_iter)}   Loss : {loss.item()}   Accuracy : {accuracy.item()}   F1 Score : {f1_score(pred_classes.detach().numpy(), Y.detach().numpy())}" )
            batch_count += 1

        return epoch_loss/len(train_iter), epoch_acc/len(train_iter)

    @staticmethod
    def evaluate_network(network, val_test_iter, optimizer, loss_fn):  
        total_loss = 0  # total loss for the whole incoming data
        total_acc = 0 # total accuracy for the whole data

        network.eval() # set the model in evaluation mode to not compute gradients and reduce overhead

        with torch.no_grad(): # turn of gradients calculation 

            for batch in val_test_iter:
                X = batch[0]
                Y = batch[1].to(torch.float32)

                predictions = network(X)#.squeeze(1) # squeeze out the extra dimension [batch_size,1]

                loss = loss_fn(predictions, Y.unsqueeze(1)) # calculate loss on the whole batch

                pred_classes = torch.round(torch.sigmoid(predictions))[:, 0]
                correct_preds = (pred_classes == Y).float()
                # get a floating tensors of predicted classes  which match original true class 

                accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]


                total_loss += loss.item() 
                total_acc += accuracy.item()

                print(f"Validation F1 Score : {f1_score(pred_classes.detach().numpy(), Y.detach().numpy())}" )


            return total_loss/len(val_test_iter), total_acc/len(val_test_iter)
    
    def Train(self, X_train, X_val, y_train, y_val, config):
        train_loader, val_loader = self.get_dataloaders(X_train, X_val, y_train, 
                                                        y_val, config.batch_size)  # setting dataloaders

        tokenizer = self.load_tokenizer(config.tokenizer_path_final)
        VOCAB_SIZE = len(tokenizer.word_index)+1  
        
        model = Network(in_neuron = VOCAB_SIZE, num_layers=config.num_layers, 
                        embedding_dim=config.embedding_dim, hidden_size=config.hidden_size, 
                        out_neuron=config.out_neuron, drop=config.drop)  # Initialising the model
        
        optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)  # Initialising the optimizer
        loss_fn = nn.BCEWithLogitsLoss()  # Initialising the loss function
        log = []
        os.makedirs(config.path_2_save_model, exist_ok=True)
        os.makedirs(config.path_2_save_logFile, exist_ok=True)
        
        for epoch in range(0, config.num_epoch):
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer, loss_fn, epoch+1)  # train for one epoch
            
            val_loss,val_acc = self.evaluate_network(model, val_loader, optimizer,loss_fn)  # test on validation dataset            
            text = f'''#End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%'''
            log.append(text)
            tqdm.write(text)
            
            if config.checkpoint_interval != -1 and epoch % config.checkpoint_interval == 0:
                torch.save(model.state_dict(), config.path_2_save_model + "model_%d.pth" % epoch)  # Save model checkpoints

        textfile = open(config.path_2_save_logFile +"log.txt", "w")  # saving the training log file
        for element in log:
            textfile.write(element + "\n")
        textfile.close() 
        torch.save(model.state_dict(), config.path_2_save_model + "model_%d.pth" % epoch)         
        return model
