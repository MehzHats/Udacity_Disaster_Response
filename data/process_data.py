import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages & categories datasets
    &
    Merge messages & categories datasets

    inputs:
    messages_filepath: string. Message dataset filepath for csv file.
    categories_filepath: string. Categories dataset filepath for csv file.

    outputs:
    df: dataframe. Dataframe containing merged content of messages & categories datasets.
    """
    #Load Messages Dataset
    messages = pd.read_csv(messages_filepath)

    #Load Categories Dataset
    categories = pd.read_csv(categories_filepath)

    #Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])

    return df


def clean_data(df):
    """
    Clean data --> remove duplicates & convert categories from strings to numeric values.

    Arguments:
    df: dataframe. df contains merged messages & categories datasets.

    Returns:
    df: dataframe. df with cleaned version of input dataframe.

    """

    # create a dataframe of the individual category columns
    categories = df["categories"].str.split(';', expand = True)

    # select first row of categories df
    row = categories.iloc[0,:]

    # extract list of categories using the row and create a new column
    cat_col_names = row.apply(lambda x: x[:-2])   #.tolist()

    # name categories column to new cols of categories
    categories.columns = cat_col_names

    # Convert category values to just numbers 0 or 1.
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]  # categories[column] = categories[column].transform(lambda x: x[-1:]

    # convert column from string to numeric/ categories[column] = categories[column].astype(int)
    categories[column] = pd.to_numeric(categories[column])
    categories.drop(categories[categories["related"] == "2"].index, inplace=True)
    categories = categories.replace({ "0": 0, "1": 1 })

    # drop the categories column from the df
    df.drop('categories', axis=1, inplace = True)

    # Concatenate   dataframe with the new `categories`
    df = pd.concat([df, categories], axis = 1)

    # Drop duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_filename):
    """
    Save df to SQLite database.

    inputs:
    df: dataframe. cleaned and merged messages and categories df.
    database_filename: path with saved sql file.

    outputs:
    None
    """

    engine = create_engine('sqlite:///'+database_filename)

    df.to_sql("Disaster_Messages", engine, index=False)

    print("Data is saved to {} in the {} table".format(database_filename, "Disaster_Messages"))


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