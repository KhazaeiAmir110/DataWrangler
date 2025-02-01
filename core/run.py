import logging

from core.test import PhonePricePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    source_csv_path = "../input_files/source.csv"
    result_csv_path = "../output_files/result.csv"
    predictor = PhonePricePredictor(source_csv_path, result_csv_path)
    is_valid, mean_error, std_error, process_time = predictor.run()
    if is_valid:
        print(f"Process completed successfully in {process_time} seconds.")
    else:
        print("Process failed.")
