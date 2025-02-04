import logging
from datetime import datetime
from typing import Tuple, Optional, Any

import pandas as pd
from pandas import DataFrame

from core.services.errors import CustomError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PhonePricePredictor:
    def __init__(self, market_posts_path, phone_main_path, phone_models_path, reference_path, result_csv_path):
        self.market_posts_path = market_posts_path
        self.phone_main_path = phone_main_path
        self.phone_models_path = phone_models_path
        self.reference_path = reference_path

        self.result_csv_path = result_csv_path

        self.data = None
        self.cleaned_data = None
        self.final_data = None

    def load_data(self) -> tuple[DataFrame | Any, DataFrame | Any, DataFrame | Any, DataFrame | Any]:
        try:
            self.market_posts_path = pd.read_csv(self.market_posts_path)
            self.phone_main_path = pd.read_csv(self.phone_main_path)
            self.phone_models_path = pd.read_csv(self.phone_models_path)
            self.reference_path = pd.read_csv(self.reference_path)
            logging.info("Data loaded successfully.")
            return self.market_posts_path, self.phone_main_path, self.phone_models_path, self.reference_path
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomError("Failed to load data.")

    def clean_data(self) -> Optional[pd.DataFrame]:
        if self.market_posts_path is None:
            raise CustomError("No data to clean.")

        try:
            self.market_posts_path = self.market_posts_path[
                (self.market_posts_path['is_original'] == True) &
                (self.market_posts_path['direct_sale'] == True) &
                self.market_posts_path['phone_id'].notna() &
                self.market_posts_path['phone_model_id'].notna()
                ]

            self.reference_path = self.reference_path[self.reference_path['is important ?'] != 'ignore']

            self.reference_path['lower bound'] = self.reference_path['lower bound'].str.replace(',', '').astype(float)
            self.reference_path['upper bound'] = self.reference_path['upper bound'].str.replace(',', '').astype(float)

            self.reference_path['mean_real'] = (self.reference_path['lower bound'] + self.reference_path[
                'upper bound']) / 2

            self.reference_path = self.reference_path.merge(self.phone_main_path, how='inner', on='nickname')
            self.reference_path = self.reference_path.merge(self.phone_models_path, how='inner',
                                                            on=['phone_id', 'internal_memory', 'ram'])

            self.reference_path = self.reference_path.rename(columns={'id': 'phone_model_id'})
            self.reference_path = self.reference_path[
                ['phone_model_id', 'nickname', 'ram', 'internal_memory', 'phone_id', 'mean_real']]

            self.reference_path['phone_id'] = self.reference_path['phone_id'].astype(float)
            self.reference_path['phone_model_id'] = self.reference_path['phone_model_id'].astype(float)

            # Debuged by alireza and amirhossein
            self.cleaned_data = self.market_posts_path.merge(self.reference_path, how='inner',
                                                             on=['phone_id', 'phone_model_id'])

            self.cleaned_data = self.cleaned_data[
                ['nickname', 'ram_y', 'internal_memory_y', 'phone_id', 'phone_model_id', 'price', 'approval_status',
                 'sub_phone_id', 'description', 'mean_real']]
            self.cleaned_data = self.cleaned_data.rename(
                columns={'internal_memory_y': 'internal_memory', 'ram_y': 'ram'})

            logging.info("Data cleaned successfully.")
            return self.cleaned_data
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise CustomError("Failed to clean data.")

    def handle_invalid_data(self) -> Optional[pd.DataFrame]:
        if self.cleaned_data is None:
            raise CustomError("No cleaned data to process.")

        try:
            mean_prices = self.cleaned_data.groupby('phone_model_id')['price'].mean().reset_index()
            mean_prices.columns = ['phone_model_id', 'mean_price']

            self.cleaned_data = self.cleaned_data.merge(mean_prices, on='phone_model_id')
            self.cleaned_data['lower_bound'] = (self.cleaned_data['mean_price'] * 0.9).round(0)
            self.cleaned_data['upper_bound'] = (self.cleaned_data['mean_price'] * 1.2).round(0)

            self.cleaned_data = self.cleaned_data[(self.cleaned_data['price'] >= self.cleaned_data['lower_bound']) & (
                    self.cleaned_data['price'] <= self.cleaned_data['upper_bound'])]
            self.cleaned_data = self.cleaned_data.drop(columns=['mean_price', 'lower_bound', 'upper_bound'])

            mean_prices = self.cleaned_data.groupby('phone_model_id')['price'].mean().reset_index()
            mean_prices.columns = ['phone_model_id', 'mean_price']

            self.final_data = self.cleaned_data.merge(mean_prices, on='phone_model_id')

            self.final_data['mean_price'] = self.final_data['mean_price'].round(0)
            self.final_data['lower_bound'] = (self.final_data['mean_price'] * 0.9).round(0)
            self.final_data['upper_bound'] = (self.final_data['mean_price'] * 1.1).round(0)

            logging.info("Invalid data handled successfully.")
            return self.final_data
        except Exception as e:
            logging.error(f"Error handling invalid data: {e}")
            raise CustomError("Failed to handle invalid data.")

    def validate_results(self) -> Tuple[bool, float, float]:
        """اعتبارسنجی نتایج"""
        if self.final_data is None:
            raise CustomError("No final data to validate.")

        try:
            # مثال: محاسبه خطا و انحراف معیار خطاها
            self.final_data['error'] = abs(self.final_data['predicted_price'] - self.final_data['actual_price']) / \
                                       self.final_data['actual_price']
            mean_error = self.final_data['error'].mean()
            std_error = self.final_data['error'].std()

            if mean_error > 0.2 or std_error > 0.1:
                logging.warning(f"Validation failed: Mean error = {mean_error}, STD error = {std_error}")
                return False, mean_error, std_error
            else:
                logging.info(f"Validation passed: Mean error = {mean_error}, STD error = {std_error}")
                return True, mean_error, std_error
        except Exception as e:
            logging.error(f"Error validating results: {e}")
            raise CustomError("Failed to validate results.")

    def save_results(self) -> None:
        """ذخیره‌سازی نتایج در فایل CSV"""
        if self.final_data is None:
            raise CustomError("No final data to save.")

        try:
            self.final_data.to_csv(self.result_csv_path, index=False)
            logging.info("Results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise CustomError("Failed to save results.")

    def run(self) -> Tuple[bool, float, float]:
        start_time = datetime.now()
        try:
            self.load_data()
            self.clean_data()
            self.handle_invalid_data()
            is_valid, mean_error, std_error = self.validate_results()
            self.save_results()
            end_time = datetime.now()
            process_time = (end_time - start_time).total_seconds()
            logging.info(f"Process completed in {process_time} seconds.")
            return is_valid, mean_error, std_error, process_time
        except CustomError as e:
            logging.error(f"Process failed: {e}")
            return False, 0, 0, 0
