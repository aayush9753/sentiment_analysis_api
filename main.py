from dataset import DataTraining
from training import Training
from inference import predict
from train_configs import config as train_config

X_train, X_val, y_train, y_val = DataTraining().get_splitted_token_vectors(train_config)  # create train and validation datasets

model = Training().Train(X_train, X_val, y_train, y_val, train_config)  # train the model and save the checkpoints according to train_configs
model.eval()

text = "This is a Good day"
print(predict(text))  # test