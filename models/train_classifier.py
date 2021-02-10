import sys
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from Wordcount import WordCounter
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    
    """Loads the data from the database and splits it into arrays for the X and Y variables.
    
    Args:
        database_filepath (string): filepath for the database that contains the message data
        
    Returns:
        X (array): x-variable data
        Y (array): y-variable data
        category_names (Index): column names that indicate the different types of message categories
    """
    
    # load the data and split into arrays for the X and Y variables
    df = pd.read_sql_table('disaster_response', 'sqlite:///'+database_filepath)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
    
    return X, Y, category_names

def tokenize(text):
    
    """Splits the text into tokens.
    
    Args:
        text (string): text to split into tokens
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
    
    # split text into tokens
    tokens = word_tokenize(text)
    
    # remove tokens that are stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # lemmatize and stem the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def build_model():
    
    """Builds model that will classify message categories.
    
    Args:
        None
    
    Returns:
        cv (object): GridSearchCV object using the defined Pipeline and parameters
    """
    
    pipeline = Pipeline([
    
        ('features', FeatureUnion([
            ('tfidf_vect', TfidfVectorizer(tokenizer=tokenize)),
            ('count', WordCounter()),
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100], 
                  'clf__estimator__criterion': ['gini', 'entropy']
                 }
    
    cv = GridSearchCV(pipeline, parameters, scoring='f1_weighted')
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """Uses the fitted model to generate predicted values, compares those against the test data, and prints out classification metrics.
    
    Args:
        model (object): fitted model to classify message categories
        X_test (array): x-variable test data
        Y_test (array): y-variable test data
        category_names (Index): column names that indicate the different types of message categories
    """
    
    Y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    
    """Exports the model as a pickle file.
    
    Args:
        model (object): fitted model to classify message categories
        model_filepath (string): filepath that stores the model
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    
    """Loads the data, builds and trains the model using the training data, uses the test data to evaluate the model, and saves the model.
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()