from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd

class WordCounter(BaseEstimator, TransformerMixin):
    
    """Class to create a transformer that counts the number of words in a message.
    """
    
    def count_words(self, text):
        
        """Counts the number of words in a given piece of text.
        
        Args:
            text (string): text with the number of words to be counted
            
        Returns:
            num_words (int): number of words in the text
        """
        
        # use regex to find detect URLs and replace them with placeholders that represent URLs
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_regex, text)
        for url in urls:
            text = text.replace(url, 'urlplaceholder')
        
        # use regex to remove punctuation (non-alphanumeric characters)
        text = re.sub(r'[^\w]', ' ', text)
        
        # normalize text
        text = text.lower().strip()
        
        # split text into tokens and count the number of tokens
        tokens = word_tokenize(text)
        num_words = len(tokens)
        
        return num_words
                   
    def fit(self, x, y=None):
        
        """Fits the estimator to the provided data.
        
        Args:
            x (array): x-variable data
            y (array): y-variable data
            
        Returns:
            self
        """
        
        return self
    
    def transform(self, X):
        
        """Transforms the data from messages to word counts.
        
        Args:
            X (array): x-variable data with messages
            
        Returns:
            X (array): x-variable data with word counts
        """
        
        # convert array to DataFrame so that the apply method can be used
        X = pd.DataFrame(X)
        
        # apply the count_words function to each column of the data
        for column in X:
            X[column] = X[column].apply(self.count_words)
            
        # convert the DataFrame back to an array
        X = X.values
            
        return X