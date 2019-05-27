# word2vec
Implementation of word2vec from scratch using Numpy

Author: Hang LE

Email: hangtp.le@gmail.com

For further details, please check out my blog post of [Understanding Word Vectors and Implementing Skip-gram with Negative Sampling](https://hangle.fr/post/word2vec/).

Note: currently only skip-gram with negative sampling is implemented. CBOW and more advanced features will be added in the future.

The code is run in the terminal using the following syntax.

## Train the model
In the train mode, running the below line will train using the training corpus and save
the model to <path_to_save_model>.

```
python skipGramNS.py --text <path_to_training_corpus> 
                     --model <path_to_save_model> 
                     --nEmbed <embedding_dimension> 
                     --negativeRate <negative_rate> 
                     --winSize <window_size> 
                     --minCount <minimum_count>
                     --stepsize <learning_rate> 
                     --epochs <number_of_epochs>
```
In which there is only 2 required arguments, which are '--text' and '--model'. The other
arguments are used to run different experiments and to save the model based on names of the
parameters to avoid overlapping. These arguments are set using default values.

I also set early stopping with the patience parameter to stop training if the loss does not 
decrease after a specified number of consecutive epochs.

## Test the model
In the test mode, running the below line will print the cosine similarity computed by the 
saved model between pairs of words in the test file.

```
python skipGram.py --text <path_to_test_corpus> 
                   --model <path_to_saved_model> 
                   --test
```

## Evaluate the model
In the evaluation mode, running the below line will compute the correlation between the cosine
similarity computed by the saved model and the ground-truth similarity. Please note that the
cosine similarity computed by the model is in the range [-1, 1].

```
python skipGram.py --text <path_to_test_corpus> 
                   --model <path_to_saved_model> 
                   --validate
```
