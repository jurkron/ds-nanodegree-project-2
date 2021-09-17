import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from given input paths into a dataframe.

    INPUT
    messages_filepath: path to a csv file that contains all messages
    categories_filepath: path to a csv file that contains all categories

    OUTPUT
    Returns a single pandas DataFrame, 
    which contains all messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on=['id'])


def clean_data(df):
    '''
        Cleans the given dataframe and removes duplicates.

        INPUT:
        df pandas.DataFrame 

        OUTPUT
        Return a cleaned dataframe. 
    '''
    categories = df['categories'].str.split(';', expand=True)
    
    # rename columns
    categories.columns = categories.iloc[0].str.replace(r'[-](.*)', '')
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(column+'-', '')
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace 'old' categories column with new categories dataframe
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Saves given DataFrame into given SQLite database file.

    INPUT
    df: pandas.DataFrame to save
    database_filename: name of the SQLite database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()