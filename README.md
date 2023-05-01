# Introduction:
This repository contains the pre-processing, model training, and hyperparameter tuning codes for the LSTM and ABLSTM models on the JPC3A dataset.

## Datasets:
In order to train or run any of the files in this model repository, the corresponding data files must be located in your directory. 

### JPC3A Dataset:
Due to the size of the JPC3A dataset, it has not been provided in the repositories. However, the dataset can be accessed by contacting Julie McCann at Imperial College London. 

## Raw Data Preprocessing:
Prior to running any of the individual files in this repository, the raw data preprocessing files located in “collected_data_preprocessing” repository (https://gitlab.doc.ic.ac.uk/g22mai03/collected_data_preprocessing) must be run on the JPC3A dataset. Please ensure that any data paths defined in the python files are changed to be compatible with your local directory. A more detailed description on running these files can be found in this repository’s README file. 

### JPC3A Raw Data Preprocessing:
To run the raw data preprocessing on the JPC3A dataset, run the respective python files:
* 1 Person Clean: “process_clean_data_1.py”
* 2 Person Clean: “process_clean_data_2.py”
* 1 Person Noisy: “process_noisy_data_1.py”
* 2 Person Noisy: “process_noisy_data_2.py”

## Preprocessing:
The "fixed_length_preprocessing.py" file must be run on each of the sub-datasets of the JPC3A dataset, changing the input and output file paths in lines 9 and 31 depending on the target sub-dataset (1 Person Clean, 2 Person Clean, 1 Person Noisy or 2 Person Noisy).

## Model Training:

### LSTM:

The "lstm_final.py" file will train the LSTM model on the full JPC3A dataset. The best hyperparameter settings for the combined model have been saved in this file. The "lstm_training_1p.py" and "lstm_training_2p.py" files can be used to train the LSTM model on the sub-datasets. These files are configured with the best hyperparameter settings and input data for the noisy datasets. The hyperparameters can be changed in lines 25-32 and the input data can be changed to the clean datasets in line 41. The test loss and accuracy per epoch, as well as the final accuracy and confusion matrices will be printed to the terminal. If "logging" in line 73 for "lstm_final.py" and line 18 for "lstm_training_1p.py" and "lstm_training_2p.py" is set to "True", this information will also be saved to the text file specified in line 74/19.

### ABLSTM:
The "ablstm_final.py" file will train the ABLSTM model on the full JPC3A dataset. The best hyperparameter settings for the combined model have been saved in this file. The "ablstm_training_1p_mha.py", "ablstm_training_1p_simple.py", "ablstm_training_2p_mha.py" and "ablstm_training_2p_simple.py" files can be used to train the LSTM model on the sub-datasets. These files are configured with the best hyperparameter settings and input data for the noisy datasets. The hyperparameters can be changed in lines 23-32 and the input data can be changed to the clean datasets in line 43, 43, 42 and 42 respectively.  The test loss and accuracy per epoch, as well as the final accuracy and confusion matrices will be printed to the terminal. If "logging" in line 76 for "lstm_final.py" and line 34 for "ablstm_training_1p_mha.py", "ablstm_training_1p_simple.py", "ablstm_training_2p_mha.py" and "ablstm_training_2p_simple.py" is set to "True", this information will also be saved to the text file specified in line 74/20.

## Hyperparameter Tuning:

### LSTM:

The "lstm_training_combined.py" file can be used for hyperparameter tuning of the LSTM model on the full JPC3A dataset. The hyperparameters can be changed in lines 25-32. The "lstm_training_1p.py" and "lstm_training_2p.py" files can be used for hyperparameter tuning on the sub-datasets. These files are currently configured to run on the noisy datasets. The hyperparameters can be changed in lines 25-32 and the input data can be changed to the clean datasets in line 41. For the "lstm_training_1p.py" and "lstm_training_2p.py" files, the validation set should be changed from the held-out test set to the internal validation set in lines 48-55 if performing hyperparameter tuning. The validation loss and accuracy per epoch, as well as the final accuracy and confusion matrices will be printed to the terminal. If "logging" in line 18 is set to "True", this information will also be saved to the text file specified in line 19.

### ABLSTM:
The "ablstm_training_combined_simple.py" and "ablstm_training_combined_mha.py" files can be used for hyperparameter tuning of the LSTM model on the full JPC3A dataset. The hyperparameters can be changed in lines 24-33. The "ablstm_training_1p_mha.py", "ablstm_training_1p_simple.py", "ablstm_training_2p_mha.py" and "ablstm_training_2p_simple.py" files can be used for hyperparameter tuning on the sub-datasets. These files are currently configured to run on the noisy datasets. The hyperparameters can be changed in lines 23-32 and the input data can be changed to the clean datasets in line 43, 43, 42 and 42 respectively. For the "ablstm_training_1p_mha.py", "ablstm_training_1p_simple.py", "ablstm_training_2p_mha.py" and "ablstm_training_2p_simple.py" files, the validation set should be changed from the held-out test set to the internal validation set in lines 56-63 if performing hyperparameter tuning. The validation loss and accuracy per epoch, as well as the final accuracy and confusion matrices will be printed to the terminal. If "logging" in line 34 is set to "True", this information will also be saved to the text file specified in line 74/20.

## Archives:
Old code files and logs have been stored in the archives folder of this repository for record-purposes.
