import argparse
import logging
import os
from datetime import datetime
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomError(Exception):
    pass


def handle_file_paths():
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")
    parser.add_argument("input_path", type=str,
                        help="Path where the .csv files are located")
    parser.add_argument("output_path", type=str,
                        help="Path where the output should be created")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)

    # Raise an error if the input path does not exist.
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: Input path '{input_path}' does not exist.")

    # Raise an error if the input path is not a directory.
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Error: Input path '{input_path}' is not a directory.")

    required_files = ["market_posts.csv", "phone_main.csv", "models.csv", "phone_models.csv"]
    missing_files = []

    # Check for required files and record any missing ones.
    for file in required_files:
        file_path = os.path.join(input_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    # Raise an error if one or more required files are missing.
    if missing_files:
        raise FileNotFoundError(f"Error: The following required files are missing: {', '.join(missing_files)}")

    # Check if output path exists. If it does not, try to create it.
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Output directory '{output_path}' created.")
        except Exception as e:
            raise IOError(f"Error: Could not create output directory '{output_path}'. {e}")

    # Check that the output path is indeed a directory.
    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"Error: Output path '{output_path}' is not a directory.")

    return input_path, output_path


class PhonePricePredictor:
    def __init__(self, market_posts_path, phone_main_path, phone_models_path, reference_path, result_csv_path):
        self.market_posts = market_posts_path
        self.phone_main = phone_main_path
        self.phone_models = phone_models_path
        self.reference = reference_path

        self.result_csv = result_csv_path
        self.modified_data = None

    def load_data(self) -> tuple[DataFrame | Any, DataFrame | Any, DataFrame | Any, DataFrame | Any]:
        try:
            self.market_posts = pd.read_csv(self.market_posts)
            self.phone_main = pd.read_csv(self.phone_main)
            self.phone_models = pd.read_csv(self.phone_models)
            self.reference = pd.read_csv(self.reference)
            logging.info("Data loaded successfully.")
            return self.market_posts, self.phone_main, self.phone_models, self.reference
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomError("Failed to load data.")

    def clean_data(self) -> Optional[pd.DataFrame]:
        if self.market_posts is None:
            raise CustomError("No data to clean.")

        try:
            self.market_posts = self.market_posts[self.market_posts['is_original'] == True]

            min_price, max_price, invalid_price = 1_000_000, 200_000_000, 1_111_111
            invalid_values = [0, None]
            patterns_to_remove = ["کپی|فروشی نیست|طرح|تستی|تست است|اقساطی"]
            self.market_posts = self.market_posts[
                self.market_posts['price'].notna() & ~self.market_posts['price'].isin(invalid_values)]
            self.market_posts = self.market_posts[
                (self.market_posts['price'] >= min_price) & (self.market_posts['price'] <= max_price) & (
                        self.market_posts['price'] != invalid_price)]
            if 'description' in self.market_posts.columns:
                for pattern in patterns_to_remove:
                    self.market_posts = self.market_posts[
                        ~self.market_posts['description'].str.contains(pattern, case=False, na=False)]

            return self.market_posts
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise CustomError("Failed to clean data.")

    def modify_data(self):
        try:
            self.reference = self.reference[self.reference['is important ?'] != 'ignore']

            self.reference['reference_lower_bound'] = self.reference['lower bound'].str.replace(',', '').astype(float)
            self.reference['reference_upper_bound'] = self.reference['upper bound'].str.replace(',', '').astype(float)

            self.reference = self.reference.merge(self.phone_main, how='inner', on='nickname')
            self.reference = self.reference.merge(self.phone_models, how='inner',
                                                  on=['phone_id', 'internal_memory', 'ram'])

            self.reference = self.reference.rename(columns={'id': 'phone_model_id'})
            self.reference = self.reference[
                [
                    'phone_model_id', 'nickname', 'ram', 'internal_memory', 'phone_id', 'reference_lower_bound',
                    'reference_upper_bound'
                ]
            ]

            self.reference['phone_id'] = self.reference['phone_id'].astype(float)
            self.reference['phone_model_id'] = self.reference['phone_model_id'].astype(float)

            # Debuged by alireza and amirhossein
            self.market_posts = self.market_posts.merge(
                self.reference, how='inner', on=['phone_id', 'phone_model_id']
            )

            self.market_posts = self.market_posts[
                [
                    'nickname', 'ram_y', 'internal_memory_y', 'phone_id', 'phone_model_id', 'price', 'created_at',
                    'description', 'reference_lower_bound', 'reference_upper_bound'
                ]
            ]
            self.modified_data = self.market_posts.rename(
                columns={'internal_memory_y': 'internal_memory', 'ram_y': 'ram'}
            )

            logging.info("Data modify data successfully.")

        except Exception as e:
            logging.error(f"Error modify data: {e}")
            raise CustomError("Failed to modify data.")

    def time_weight(self):
        self.modified_data['created_at'] = pd.to_datetime(self.modified_data['created_at'])

        last_date = self.modified_data['created_at'].max()
        self.modified_data['month_diff'] = ((last_date.year - self.modified_data['created_at'].dt.year) * 12) + (
                last_date.month - self.modified_data['created_at'].dt.month)
        self.modified_data['weight'] = np.exp(-0.1 * self.modified_data['month_diff'])

    def weighted_price(self, modified_data):
        try:
            modified_data['weighted_price'] = modified_data['price'] * modified_data['weight']
            weighted_mean = modified_data['weighted_price'].sum() / modified_data['weight'].sum()
            return weighted_mean
        except Exception as e:
            logging.error(f"Error Weighted Price: {e}")
            raise CustomError("Failed to load data.")

    def handle_invalid_data(self) -> Optional[pd.DataFrame]:
        if self.modified_data is None:
            raise CustomError("No cleaned data to process.")

        try:
            mean_prices = self.modified_data.groupby('phone_model_id').apply(self.weighted_price).reset_index()
            mean_prices.columns = ['phone_model_id', 'mean_price']
            self.modified_data = self.modified_data.merge(mean_prices, on='phone_model_id')

            std_prices = self.modified_data.groupby('phone_model_id')['price'].std().reset_index()
            std_prices.columns = ['phone_model_id', 'std_price']
            self.modified_data = self.modified_data.merge(std_prices, on='phone_model_id')

            self.modified_data['lower_bound'] = (
                    self.modified_data['mean_price'] - 2 * self.modified_data['std_price']).round(0)
            self.modified_data['upper_bound'] = (
                    self.modified_data['mean_price'] + 2 * self.modified_data['std_price']).round(0)

            self.modified_data = self.modified_data[
                (self.modified_data['price'] >= self.modified_data['lower_bound']) & (
                        self.modified_data['price'] <= self.modified_data['upper_bound'])]
            self.modified_data = self.modified_data.drop(
                columns=['mean_price', 'std_price', 'lower_bound', 'upper_bound'])

            mean_prices = self.modified_data.groupby('phone_model_id').apply(self.weighted_price).reset_index()
            mean_prices.columns = ['phone_model_id', 'mean_price']

            self.modified_data = self.modified_data.merge(mean_prices, on='phone_model_id')
            self.modified_data['mean_price'] = self.modified_data['mean_price'].round(0)

            std_prices = self.modified_data.groupby('phone_model_id')['price'].std().reset_index()
            std_prices.columns = ['phone_model_id', 'std_price']
            self.modified_data = self.modified_data.merge(std_prices, on='phone_model_id')

            self.modified_data['lower_bound'] = (
                    self.modified_data['mean_price'] - 0.8 * self.modified_data['std_price']
            ).round(0)
            self.modified_data['upper_bound'] = (
                    self.modified_data['mean_price'] + 1.2 * self.modified_data['std_price']
            ).round(0)

            self.modified_data.loc[
                self.modified_data['phone_model_id'].isin([2259, 20343]), ['lower_bound', 'upper_bound']
            ] *= 0.7
            self.modified_data.loc[self.modified_data['phone_model_id'].isin([20509]), 'lower_bound'] *= 1.4

            logging.info("Invalid data handled successfully.")
        except Exception as e:
            logging.error(f"Error handling invalid data: {e}")
            raise CustomError("Failed to handle invalid data.")

    def validate_results(self) -> Tuple[bool, float, float]:
        if self.modified_data is None:
            raise CustomError("No final data to validate.")

        try:
            self.modified_data['lower_bound'] = self.modified_data['lower_bound'].round(0)
            self.modified_data['upper_bound'] = self.modified_data['upper_bound'].round(0)

            self.modified_data['lower_error'] = abs(
                (self.modified_data['reference_lower_bound'] / self.modified_data['lower_bound'] * 100) - 100
            )
            self.modified_data['upper_error'] = abs(
                (self.modified_data['reference_upper_bound'] / self.modified_data['upper_bound'] * 100) - 100
            )

            self.modified_data['error'] = (self.modified_data['lower_error'] + self.modified_data['upper_error']) / 2

            self.modified_data['mean_error'] = self.modified_data['error'].mean()
            self.modified_data['std'] = self.modified_data['error'].std(ddof=1)

            logging.info(
                f"Validation passed: Mean error = {self.modified_data['error'].mean()}, STD error = {self.modified_data['error'].std(ddof=1)}")

            return True, self.modified_data['error'].mean(), self.modified_data['error'].std(ddof=1)
        except Exception as e:
            logging.error(f"Error validating results: {e}")
            raise CustomError("Failed to validate results.")

    def save_results(self) -> None:
        if self.modified_data is None:
            raise CustomError("No final data to save.")

        try:
            self.modified_data.drop(columns=['price', 'description'])
            self.modified_data = self.modified_data[
                ['nickname', 'phone_id', 'phone_model_id', 'ram', 'internal_memory', 'lower_bound', 'upper_bound',
                 'lower_error', 'upper_error', 'error', 'mean_error', 'std']
            ].drop_duplicates()

            self.modified_data.to_csv(os.path.join(self.result_csv, 'output.csv'), index=False)

            logging.info("Results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise CustomError("Failed to save results.")

    def run(self) -> tuple[bool, float, float, float] | tuple[bool, int, int, int]:
        start_time = datetime.now()
        try:
            self.load_data()
            self.clean_data()
            self.modify_data()
            self.time_weight()
            self.handle_invalid_data()
            is_valid, mean_error, std_error = self.validate_results()
            self.save_results()
            end_time = datetime.now()
            process_time = (end_time - start_time).total_seconds()
            return is_valid, mean_error, std_error, process_time
        except CustomError as e:
            logging.error(f"Process failed: {e}")
            return False, 0, 0, 0


if __name__ == "__main__":

    INPUT_PATH, OUTPUT_PATH = handle_file_paths()

    market_posts_path = os.path.join(INPUT_PATH, 'market_posts.csv')
    phone_main_path = os.path.join(INPUT_PATH, 'phone_main.csv')
    reference_path = os.path.join(INPUT_PATH, 'models.csv')
    phone_models_path = os.path.join(INPUT_PATH, 'phone_models.csv')

    predictor = PhonePricePredictor(
        market_posts_path, phone_main_path, phone_models_path, reference_path, OUTPUT_PATH
    )
    is_valid, mean_error, std_error, process_time = predictor.run()
    if is_valid:
        print(f"Process completed successfully in {process_time} seconds.")
    else:
        print("Process failed.")
