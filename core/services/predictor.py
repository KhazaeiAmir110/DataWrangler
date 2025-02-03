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
        try:
            self.data = pd.read_csv(self.source_csv_path)
            logging.info("Data loaded successfully.")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomError("Failed to load data.")

    def run(self) -> tuple[bool, float, float, float] | tuple[bool, int, int, int]:
        start_time = datetime.now()
        try:
            self.load_data()
            # self.clean_data()
            # self.save_results()
            end_time = datetime.now()
            process_time = (end_time - start_time).total_seconds()
            logging.info(f"Process completed in {process_time} seconds.")
            return process_time
        except CustomError as e:
            logging.error(f"Process failed: {e}")
            return False, 0, 0, 0
