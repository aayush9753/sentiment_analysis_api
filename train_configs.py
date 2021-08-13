class config:
    data_path = 'airline_sentiment_analysis.csv'
    val_size = 0.2
    random_state=259
    max_pad_len = 25
    default_tokenizer = True
    tokenizer_path = 'tokenizer.pickle'
    save_tokenizer_to ='tokenizer.pickle'
    tokenizer_path_final = 'tokenizer.pickle'
    
    
    num_epoch = 50
    batch_size = 4096
    path_2_save_model = "saved/"
    path_2_save_logFile = "saved/"
    checkpoint_interval = 3
    lr=0.001
    
    # Model Parameters
    num_layers = 2
    embedding_dim=200
    hidden_size=128
    out_neuron = 1
    drop=0.4
    
    pretrained = True
    
    path_2_saved_model = "saved/model_49.pth"