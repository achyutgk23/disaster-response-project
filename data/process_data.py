import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the messages and categories datasets

    args:
    messages_filepath: str. Pathway or the location of the
                       disaster_messages.csv file
    categories_filepath: str. Pathway or the location of the
                         disaster_categories.csv file

    Returns:
    df: DataFrame. A dataframe containing both messages and categories datasets
    """

    # Load the messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load the categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge two datasets into one
    df = messages.merge(categories, on='id')

    return df



def clean_data(df):
    """Clean the data for classification algorithms

    args:
    df: DataFrame. A dataframe containing both messages and categories datasets

    Returns:
    df: DataFrame. A cleaned dataframe which will be further fed into
                   a classification algorithm
    """

    # Split categories column into 36 different columns
    categories_new = df['categories'].str.split(';', expand=True)

    # Extract column names from the categories_new dataset
    row = categories_new.loc[0]
    category_colnames = row.str.split('-').str.get(0)

    # Rename the column names of categories_new dataset
    categories_new.columns = category_colnames

    # Convert the values in each column into '0s' and '1s'
    for column in categories_new:
        categories_new[column] = categories_new[column].str.split('-').str.get(1)
        categories_new[column] = categories_new[column].astype(int)

    #replace '2s' with '1s' as there are some '2s' in the dataset
    categories_new = categories_new.replace(2, 1)

    # Replace 'categories' column in df with 'categories_new' dataset
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_new], axis=1)

    # Remove the duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save the cleaned dataset into sqlite database

    Args:
    df: DataFrame. The cleaned dataframe from clean_data() function
    database_filename: str. A database file saved by the
                       specified name in the local directory

    Returns:
    None
    """

    # Save the data into a sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse.db', engine, index=False, if_exists='replace')


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
