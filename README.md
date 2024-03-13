# NLP - News Topic Identification
-- Yuanshan Zhang, Mengxin Zhao, Yahui Wen, Yiming Wang, Jiayun Liu

**Note: due to limited time, we used test scores to evaluate our models instead of cross-validation scores, which is not a desirable practice.*

## What I did
**1. NLP**\
In this project, I applied text mining techniques to the AG news corpus to classify the news based on their categories. First, I cleaned up and converted the txt file into csv format. Second, I randomly sampled 5000 data points from 4 categories:  'Business', 'Entertainment', 'Sports', 'Sci/Tech', and split the sampled data into a train set and test set. Then, I preprocessed the text data by using regular expressions, tokenizing , removing stop words, and lemmatizing. 

After preprocessing, I applied vectorization and embedding to preprocessed documents. For vectorization, I used BoW (Bag of Words) and TF-IDF (Term Frequency - Inverse Document Frequency). For embedding, I trained a Word2Vec model and loaded a pre-trained GloVe model("glove-wiki-gigaword-100") and calculated their average, TF-IDF weighted average, and IDF-weighted average. 

**2. Supervised ML**\
After vectorizing and embedding, I compared the performance of BoW, TF-IDF, Word2Vec, and GloVe using Random Forest:

| models | test accuracy |                                     
|-------|-------|
| Bag of Words | 0.78 |
| TF-IDF | 0.78 |
| Word2Vec - average | 0.72 |
| Word2Vec - IDF weighted average | 0.73 |
| Word2Vec - TF-IDF weighted average | 0.72 |
| "glove-wiki-gigaword-100" - average | 0.80 |

Finally, I chose the most promising embedding method (i.e. GloVe) and used deep learning to enhance its performance.

**3. Deep Learning**\
The fundamental technique that lies under today’s Large Language Models is the Recurrent Neural Network. However, it is hard for RNN to keep track of early information due to gradient exploding/vanishing issues that usually happen when the sequence length, or equivalently, the time step is greater than 100. LSTM, on the other hand, partially fixes this problem by using gate control. To investigate these two models, I built two LSTMs (one with pre-train GloVe as its embedding and one with an embedding layer) and one RNN. 

![示例图片](Images/RNN-LSTM.png)

The data preparation procedures are as follows:
1. Embed each token into a fixed-size vector (n_dim << n_tokens)\
‘I’ → [0.2, 0.1], ‘am’ → [0.8, 0.63], ‘batman’ → [0.33, 0.99]

3. For each document, arrange vectors to form sequences\
[‘I’, ‘am’, ‘batman’] → [[0.2, 0.1], [0.8, 0.63], [0.33, 0.99]]

4. Pad sequences so that all documents have the same length\
[[0.2, 0.1], [0.8, 0.63], [0.33, 0.99]] → [[0.2, 0.1], [0.8, 0.63], [0.33, 0.99], [0, 0], [0, 0]]

5. One-hot encode the target\
[‘Business’, ‘Entertainment’, ‘Sci/Tech’, ‘Sports’] → [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

Further, the convolutional technique has proven its efficiency in combining with LSTM and RNN for feature extraction and dimensionality reduction. So before I passed the document matrix to the hidden layer of the LSTM and RNN, I used the convolutional and max pooling layer to perform feature extraction, reducing time steps from 180 to 44 and embedding dimension from 100 to 80.

<img src="Images/Convolution-Maxpooling.png" alt="示例图片" width="620" height="307">

The model performance is summarized as follows:
| models | test accuracy |                                     
|-------|-------|
| RNN | 0.76 |
| LSTM | 0.78 |
| LSTM + "glove-wiki-gigaword-100" | 0.83 |

## Conclusions
- Vectorization methods like Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) provide descent accuracies with traditional machine learning models, such as Random Forest, achieving test accuracies of 0.78 for both. However, these methods inevitably suffer from the curse of dimensionality due to their high-dimensional sparse representations, which may not effectively capture the semantic richness of text, leading to overfitting issues.

- Embedding techniques such as Word2Vec and GloVe offer more sophisticated solutions by encoding semantic relationships between words, which not only significantly alleviate the curse of dimensionality but also improves model performance by utilizing contextual information. For Word2Vec, using the average, IDF, or TF-IDF weighted average embedding yields similar test accuracies, indicating that simple averaging of word embeddings can be sufficient for achieving document-level vector representations in our project.

- The utilization of pre-trained GloVe embeddings notably enhances test accuracy to 0.8, highlighting the benefits of transfer learning, especially in contexts with limited data availability. Pre-trained word embeddings provide a significant language understanding basis that substantially boosts model performance.

- For deep learning models, LSTM with pre-trained GloVe embeddings achieves the highest test accuracy of 0.83, underscoring the advantage of deep learning models in learning from sequential input and understanding contextual nuances.

- Simple RNNs and LSTMs without pre-trained embeddings are prone to overfitting, exhibit lower accuracy, and converge slower. Therefore, in scenarios with smaller datasets, leveraging pre-trained embeddings emerges as an exceptional practice for accelerating convergence and enhancing performance and generalization ability on unseen data.
