# CAPTCHA_IND
This is a tool to identify CAPTCHA basing on CNN implemented by mxnet.

The cnn model should be trained before taken into use.

### get the training datasets
If you would like to download and construct the datasets from the very begining, just remove my datasets directory and run the **get_datasets.sh** script:
```
    $ cd CAPTCHA_IND 
    $ rm -rf datasets
    $ sh tools/get_datasets.sh
```
This may take a long time to download. Otherwise, you may simply use the datasets I downloaded and constructed for you by skipping this step.


