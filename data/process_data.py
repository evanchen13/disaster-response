import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """Loads messages and categories datasets and merges them into one dataframe.
    
    Args:
        messages_filepath (string): filepath for the file that contains messages data
        categories_filepath (string): filepath for the file that contains categories data
        
    Returns:
        df (DataFrame): dataset with messages and categories data
    """
    
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets using common id
    df = messages.merge(categories, how='inner', on='id')
    
    return df

def clean_data(df):
    
    """Cleans dataset by extracting category classifications and dropping duplicates.
    
    Args:
        df (DataFrame): dataset to be cleaned
    
    Returns:
        df (DataFrame): cleaned dataset
    """
    
    # create dataframe of individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # use the first row to get the categories to use as column names
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename columns
    categories.columns = category_colnames
    
    # for each column, extract numbers from the strings, replace any 2 with 1, and convert the column to numeric
    for column in categories:
        categories[column] = categories[column].str[-1:].str.replace('2', '1').astype(int)
        
    # drop the categories column from the original dataframe
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    """Saves the cleaned dataset into the specified database file.
    
    Args:
        df (DataFrame): cleaned dataset
        database_filename (string): file name for the database that will store the cleaned data
        
    Returns:
        None
    """

    # create engine and save cleaned dataframe to the database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False)

def main():
    
    """Loads, cleans, and saves the dataset.
    """
    
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