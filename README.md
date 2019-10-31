# Text-Classification
Two text classification models for detecting positive and negative movie reviews
# Dataset:
The IMDB dataset has been used for training and testing purposes.
# Dependancies
* Used Python Version:3.7.0
* Install necessary modules with `sudo pip3 install -r requiremnets.txt` command.
# Model Training and Testing:
To train and test the model --> `python3 train_and_test.py`
# Model Parameters:
For RNN:
  * Embedding_dimension = 100
  * Hidden_dimension = 256
  * Output_dimension = 1
  * Optimizer = Stochastic Gradient Descent
  * Loss criterion = Binary Cross Entropy with LogitsLoss

For CNN:
  * Embedding_dimension = 100
  * Number of filters = 100
  * Filter Sizes = [3 , 4 , 5]
  * Output_dimension = 1
  * Dropout = 0.5
  * Optimizer = Adaptive Momentum
  * Loss criterion = Binary Cross Entropy with LogitsLoss
# Author:
Subhrajit Dey(@subro608)
  
  
  
  
