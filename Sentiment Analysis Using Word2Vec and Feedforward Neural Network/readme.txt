Sentiment Analysis Using Word2Vec and Feedforward Neural Network
- This project involves building a sentiment analysis classifier using Word2Vec embeddings and a Feedforward Neural Network (FFNN). The classifier is trained on a dataset of text reviews and evaluated using various metrics.

Files and Directories
- SentimentData.csv: Dataset file containing text reviews and their sentiment labels.
- DLProject.ipynb: Jupyter notebook containing the implementation of the project.

How to Run the Code:
- Ensure the dataset file (SentimentData.csv) is placed in the same directory as the script or Jupyter notebook.
- Install the required Python packages if not already installed:
  pip install pandas numpy scikit-learn gensim tensorflow matplotlib nltk
- Open and run the provided python notebook (DLProject.ipynb) in your preferred environment (e.g., Google Colab, Jupyter Lab, Jupyter Notebook, or VS Code).

Description of the Code:
- Loads the dataset containing reviews and sentiment labels.
- Cleans the text by removing special characters, punctuation, and stopwords.
- Converts text to lowercase and applies lemmatization.
- Uses pre-trained Word2Vec embeddings to generate dense vector representations of words.
- Aggregates word embeddings into a single fixed-length vector for each review.
- Builds a Feedforward Neural Network with multiple layers and ReLU/Sigmoid activation functions.
- Splits the dataset into training and test sets.
- Trains the FFNN on the training data and validates its performance during training.
- Evaluates the model's performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Plots training and validation accuracy/loss curves.
- Displays metrics like accuracy, confusion matrix, classification report, and ROC curve.

Output:
- Training Time: The duration taken to train the model.
- Validation Accuracy: Accuracy on the validation set (optional).
- Test Accuracy: Accuracy on the test set.
- Precision and Recall: Precision and recall scores on the test set.
- Confusion Matrix: A visual representation of the confusion matrix.
- Classification Report: A detailed report including precision, recall, and F1-score.
- Training Curves: Plots showing training and validation accuracy/loss over epochs.

Authors:
- Sampath Reddy Kothakapu
