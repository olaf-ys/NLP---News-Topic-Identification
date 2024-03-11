# NLP - News Topic Identification
-- Yuanshan Zhang, Mengxin Zhao, Yahui Wen, Yiming Wang, Jiayun Liu

## What I did
In this project, I applied text mining techniques to the AG news corpus to classify the news based on their categories. First, I cleaned up and converted the txt file into csv format. Then, I preprocessed the text data by using regular expression, tokenizing the text, removing stop words, and lemmatizing the tokens. After preprocessing, I vectorized the tokens using BoW (Bag of Words), TF-IDF (Term Frequency - Inverse Document Frequency). 

For document level word vectors representations, I trained a Word2Vec model as well as utilized a pre-trained GloVe model to get word embeddings and caluculated the average, TF-IDF weighted average, and IDF weighted average of embedded vectors. Finally, I investigated the performance of these vectors on Random Forest, RNN, and LSTM.

## Deep Learning
The fundamental technique that lies under today’s Large Language Models is the Recurrent Neural Network. However, it is hard for RNN to keep track of early information due to gradient exploding/vanishing issues that usually happens when the sequence length, or equivalently, the time step is greater than 100. LSTM, one the other hand, partially fixes this problem by using gate control. To investigate these 2 models, I built 2 LSTMs (one with pre-train GloVe as its embedding and one with an embedding layer) and one RNN. 

The preprocessing procedures are as follows:
1. Embed each token into a fixed-size vector (n_dim << n_tokens)\
‘I’ → [0.2, 0.1], ‘am’ → [0.8, 0.63], ‘batman’ → [0.33, 0.99]

3. For each document, arrange vectors to form sequence\
[‘I’, ‘am’, ‘batman’] → [[0.2, 0.1], [0.8, 0.63], [0.33, 0.99]]

4. Pad sequences so that all documents have the same length\
[[0.2, 0.1], [0.8, 0.63], [0.33, 0.99]] → [[0.2, 0.1], [0.8, 0.63], [0.33, 0.99], [0, 0], [0, 0]]

5. One-hot encode the target\
[‘Business’, ‘Entertainment’, ‘Sci/Tech’, ‘Sports’] → [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

Further, the Convolutional technique has proven its efficiency in combining with LSTM and RNN. So before I pass the document matrix to the hidden layer of the LSTM and RNN, I used the convolutional and max pooling layer to perform feature extraction, reducing time steps from 180 to 44 and embedding dimension from 100 to 80.
