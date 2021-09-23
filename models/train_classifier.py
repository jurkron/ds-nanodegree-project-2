import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load dataset from given SQLite Database.

    INPUT
    database_filepath: file path to the database file

    OUTPUT
    X: Dataset containing the messages for train and test
    y: Dataset containing all categories for X used in train and test
    category_name: a list of category names, these are the column names of y
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    y = df.drop(columns=['id','message','original','genre'])
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    '''
    Tokenizes the given text.

    INPUT
    text: a string to tokenize

    OUTPUT
    clean_tokens: a cleaned lemmatized array of the given INPUT
    '''
    # text in lower case and remove all characters that are not numbers or 
    # word character
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # tokenize text
    tokens = nltk.tokenize.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Returns a model by using a pipeline with different transformers and a multi output classifier. 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
    #    'clf__estimator__min_samples_split': [2, 3, 4]
    }

    return GridSearchCV(pipeline, n_jobs=-2, param_grid=parameters, verbose=3, cv=5)


def evaluate_model(model, X_test, y_test, category_names):
    '''
    The f1 score, precision and recall for the given test dataset is printed out for each category.

    INPUT
    model: the model to test
    X_test: a dataset used to make predictions on the given model
    y_test: a dataset containing the true responses on X_test, to evaluate the predictions against
    category_names: list of the category names of y_test
    '''
    y_pred = model.predict(X_test)

    df_pred = pd.DataFrame(y_pred, columns=category_names)

    df_report = pd.DataFrame(columns=['f1-score','precision','recall'])
    for column in y_test.columns:
        report = pd.Series({
            'f1-score':f1_score(y_test[column],df_pred[column]),
            'precision':precision_score(y_test[column],df_pred[column]),
            'recall':recall_score(y_test[column],df_pred[column])
        }, name=column)
        df_report = df_report.append(report)

    print(df_report)


def save_model(model, model_filepath):
    '''
    Save the model to given filepath

    INPUT
    model: model to save
    model_filepath: the file path where the model should be saved
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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