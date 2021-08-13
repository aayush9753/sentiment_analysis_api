# Sentiment Analysis Api
- To get similar results and proper functioning of the code, extract the files in same directory as ther are in this repo.
### To Run the Fast Api : 
- **uvicorn api:app --reload** in Python shell or terminal.
-  Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/docs) for the Swagger Documentation and the fast api which you can test there only.


## Sample Outputs of FastApi    
:-------------------------:|:-------------------------:
![](https://github.com/aayush9753/sentiment_analysis_api/blob/main/repo_images/output_example.jpg)

## Training
- Run the train_configs file and set the parameter for training
-:-------------------------:|:-------------------------:
![](https://github.com/aayush9753/sentiment_analysis_api/blob/main/repo_images/train_configs.jpg)
- data_path => Path to the dataset
- val_size => Size of validation data to be used
- default_tokenizer => Whether to use the already existing tokenizer
- tokenizer_path => Path to the already existing tokenizer
- save_tokenizer_to => Path to save the new tokenizer if default_tokenizer is False
- tokenizer_path_final => path to tokenizer that you wanna use
