import logging
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd

from core.services.errors import CustomError


class PhonePricePredictor:
    def __init__(self, source_csv_path: str, result_csv_path: str):
        self.source_csv_path = source_csv_path
        self.result_csv_path = result_csv_path
        self.data = None
        self.cleaned_data = None
        self.final_data = None

    def load_data(self) -> Optional[pd.DataFrame]:
        """خواندن داده‌ها از فایل CSV"""
        try:
            self.data = pd.read_csv(self.source_csv_path)
            logging.info("Data loaded successfully.")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomError("Failed to load data.")

    def clean_data(self) -> Optional[pd.DataFrame]:
        """پاکسازی داده‌ها"""
        if self.data is None:
            raise CustomError("No data to clean.")

        try:
            # مثال: حذف ردیف‌های با مقادیر خالی
            self.cleaned_data = self.data.dropna()
            # مثال: تبدیل ستون‌ها به نوع داده مناسب
            self.cleaned_data['post_price'] = self.cleaned_data['post_price'].astype(float)
            logging.info("Data cleaned successfully.")
            return self.cleaned_data
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise CustomError("Failed to clean data.")

    def handle_invalid_data(self) -> Optional[pd.DataFrame]:
        """حذف داده‌های نامعتبر"""
        if self.cleaned_data is None:
            raise CustomError("No cleaned data to process.")

        try:
            # مثال: حذف داده‌های پرت بر اساس قیمت
            Q1 = self.cleaned_data['post_price'].quantile(0.25)
            Q3 = self.cleaned_data['post_price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.cleaned_data = self.cleaned_data[
                (self.cleaned_data['post_price'] >= lower_bound) & (self.cleaned_data['post_price'] <= upper_bound)]
            logging.info("Invalid data handled successfully.")
            return self.cleaned_data
        except Exception as e:
            logging.error(f"Error handling invalid data: {e}")
            raise CustomError("Failed to handle invalid data.")

    def modify_data(self) -> Optional[pd.DataFrame]:
        """اعمال تغییرات بر روی داده‌ها"""
        if self.cleaned_data is None:
            raise CustomError("No cleaned data to modify.")

        try:
            # مثال: اعمال یک تعدیل ساده بر روی قیمت
            self.final_data = self.cleaned_data.copy()
            self.final_data['predicted_price'] = self.final_data['post_price'] * 0.9  # کاهش 10% قیمت
            logging.info("Data modified successfully.")
            return self.final_data
        except Exception as e:
            logging.error(f"Error modifying data: {e}")
            raise CustomError("Failed to modify data.")

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
        """اجرای تمام مراحل"""
        start_time = datetime.now()
        try:
            self.load_data()
            self.clean_data()
            self.handle_invalid_data()
            self.modify_data()
            is_valid, mean_error, std_error = self.validate_results()
            self.save_results()
            end_time = datetime.now()
            process_time = (end_time - start_time).total_seconds()
            logging.info(f"Process completed in {process_time} seconds.")
            return is_valid, mean_error, std_error, process_time
        except CustomError as e:
            logging.error(f"Process failed: {e}")
            return False, 0, 0, 0
