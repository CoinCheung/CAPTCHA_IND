# CAPTCHA_IND
This is a tool to identify CAPTCHA basing on CNN implemented by mxnet.

The cnn model should be trained before taken into use.

### 1. Get the training datasets
If you would like to download and construct the datasets from the very begining, just remove my datasets directory and run the **get_datasets.sh** script:
```
    $ cd CAPTCHA_IND 
    $ sh scripts/clear.sh data
    $ sh scripts/get_datasets.sh
```
This may take a long time to download. Otherwise, you may simply use the datasets I downloaded and constructed for you by skipping this step.


### 2. A bit setup
So as to install some necessary packages, just execute the setup.sh script:
```
    $ sh scripts/setup.sh
```


### 3. Train the model
To train the model, run the train.py script:
```
    $ python train.py
```
This will train the model with datasets downloaded in step 1. Besides, the trained model will be exported and saved to the directory model_export. If one would like to remove the model saved before, he or she could run the script clear.sh like this:
```
    $ sh scripts/clear.sh model
```


### 4. Test



### 5. Random search hyper parameters 
A python script is provided to search the hyper parameters of learning_rrate, learning_rrate_factor (let lr decay each 500 iterations) together with the weight decay (L2 regularization). Run the python script and a log file will be generated in the current directory which records the model hyper parameters and their associating validation accuracy.
```
    $ python random_search_hypers.py
```
