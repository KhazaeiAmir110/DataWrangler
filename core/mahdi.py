from time import time

import pandas as pd

DATA_PATH = '../input_files/db_data/'

market_posts_path = DATA_PATH + 'market_posts.csv'
phone_main_path = DATA_PATH + 'phone_main.csv'
refrence_path = DATA_PATH + 'models.csv'
phone_models_path = DATA_PATH + 'phone_models.csv'


def error_percentage(df: pd.DataFrame):
    df['error'] = abs((df['mean_real'] / df['mean_price'] * 100) - 100)
    mean_error = df['error'].mean()
    std = df['error'].std(ddof=1)
    print('std:', std)
    print('mean_error:', mean_error)
    print(df)


def price_bounds(df: pd.DataFrame):
    mean_prices = df.groupby('phone_model_id')['price'].mean().reset_index()
    mean_prices.columns = ['phone_model_id', 'mean_price']

    df = df.merge(mean_prices, on='phone_model_id')
    df['lower_bound'] = (df['mean_price'] * 0.9).round(0)
    df['upper_bound'] = (df['mean_price'] * 1.2).round(0)

    df = df[(df['price'] >= df['lower_bound']) & (df['price'] <= df['upper_bound'])]
    df = df.drop(columns=['mean_price', 'lower_bound', 'upper_bound'])

    mean_prices = df.groupby('phone_model_id')['price'].mean().reset_index()
    mean_prices.columns = ['phone_model_id', 'mean_price']

    df = df.merge(mean_prices, on='phone_model_id')
    df['mean_price'] = df['mean_price'].round(0)
    df['lower_bound'] = (df['mean_price'] * 0.9).round(0)
    df['upper_bound'] = (df['mean_price'] * 1.1).round(0)

    error_percentage(df)


def cleaned_dataframe(df: pd.DataFrame):
    refrence = pd.read_csv(refrence_path)
    refrence = refrence[refrence['is important ?'] != 'ignore']

    refrence['lower bound'] = refrence['lower bound'].str.replace(',', '').astype(float)
    refrence['upper bound'] = refrence['upper bound'].str.replace(',', '').astype(float)

    refrence['mean_real'] = (refrence['lower bound'] + refrence['upper bound']) / 2

    phone_main = pd.read_csv(phone_main_path)
    refrence = refrence.merge(phone_main, how='inner', on='nickname')

    phone_models = pd.read_csv(phone_models_path)
    refrence = refrence.merge(phone_models, how='inner', on=['phone_id', 'internal_memory', 'ram'])

    refrence = refrence.rename(columns={'id': 'phone_model_id'})
    refrence = refrence[['phone_model_id', 'nickname', 'ram', 'internal_memory', 'phone_id', 'mean_real']]

    refrence['phone_id'] = refrence['phone_id'].astype(float)
    refrence['phone_model_id'] = refrence['phone_model_id'].astype(float)

    # Debuged by alireza and amirhossein
    df = df.merge(refrence, how='inner', on=['phone_id', 'phone_model_id'])

    df = df[['nickname', 'ram_y', 'internal_memory_y', 'phone_id', 'phone_model_id', 'price', 'approval_status',
             'sub_phone_id', 'description', 'mean_real']]
    df = df.rename(columns={'internal_memory_y': 'internal_memory', 'ram_y': 'ram'})
    price_bounds(df)


def validated_dataframe():
    market_posts = pd.read_csv(market_posts_path)
    tempdf = market_posts

    tempdf = tempdf[
        (tempdf['is_original'] == True) &
        (tempdf['direct_sale'] == True) &
        tempdf['phone_id'].notna() &
        tempdf['phone_model_id'].notna()
        ]
    cleaned_dataframe(tempdf)


start = time()
validated_dataframe()
print(time() - start)
