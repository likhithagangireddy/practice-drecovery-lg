import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath: File path of messages data
        categories_filepath: File path of categories data
    Output:
        Two dataframes messages and categories
    '''
    # Read message data
    messages = pd.read_csv(messages_filepath)
    # Read categories data
    categories = pd.read_csv(categories_filepath)

    return messages, categories


def clean_data(messages, categories):
    '''
    Input:
        messages, categories: dataframes from messages and categories
    Output:
        df: Cleaned dataset
    '''

    i = categories.columns.get_loc('categories')
    mycategories2 = categories['categories'].str.split(';',expand=True)
    categories = pd.concat([categories.iloc[:, :i], mycategories2, categories.iloc[:, i+1:]], axis=1)

    print(type(categories))
    print("About to fail")
    print(categories.columns)
    print(categories.head(2))

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    category_colnames = category_colnames.replace('', 'id')
    # rename the columns of `categories`
    categories.columns = category_colnames


    for column in categories:
        # set each value to be the last character of the string
        if(column != 'id'):
            categories[column] = categories[column].apply(lambda x : x[-1:])

            # convert column from string to numeric
            categories[column] =  categories[column].astype(int)

    df = pd.merge(messages,categories, on=['id'])
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Save df into sqlite db
    Input:
        df: cleaned dataset
        database_filename: database name, e.g. DisasterMessages.db
    Output:
        A SQLite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('labeled_messages', engine, index=False , if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df1, df2 = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df1, df2)

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
