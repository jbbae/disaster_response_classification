import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - (str) file path for database
    
    OUTPUT:
    X - (pandas dataframe) df with independent variables
    Y - (pandas dataframe) df with dependent variables
    list(Y) - (list) list with category names
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('categorized_msgs', engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, list(Y)


def tokenize(text):
    '''
    INPUT:
    text - (str) raw text message to perform lemmatization + tokenization
    
    OUTPUT:
    clean_tokens - (array) Array of lemmatized words
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
        
    return clean_tokens

def build_model():
    '''
    INPUT:
    none
    
    OUTPUT:
    pipeline - (pipeline) NLP pipeline that performs classification based on tokenized messages
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - (pipeline) NLP classification pipeline trained by "build_model"
    X_test - (pandas dataframe) df with testing set for X
    Y_test - (pandas dataframe) df with testing set for Y
    category_names - (list) list with unique category names
    
    OUTPUT:
    none
    
    '''
    y_pred = model.predict(X_test)

    for i, y_pred_col in enumerate(y_pred.transpose()):
        print(category_names[i])
        print(classification_report(Y_test[category_names[i]], y_pred_col))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

    return

def main():
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