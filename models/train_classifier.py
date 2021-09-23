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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    y = df.drop(columns=['id','message','original','genre'])
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
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
    The f1 score, precision and recall for the test set is outputted for each category.
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