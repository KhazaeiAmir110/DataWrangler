import logging

from core.services.predictor import PhonePricePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    DATA_PATH = '../input_files/db_data/'

    market_posts_path = DATA_PATH + 'market_posts.csv'
    phone_main_path = DATA_PATH + 'phone_main.csv'
    reference_path = DATA_PATH + 'models.csv'
    phone_models_path = DATA_PATH + 'phone_models.csv'

    result_csv_path = "../output_files/result.csv"
    predictor = PhonePricePredictor(
        market_posts_path, phone_main_path, phone_models_path, reference_path, result_csv_path
    )
    is_valid, mean_error, std_error, process_time = predictor.run()
    if is_valid:
        print(f"Process completed successfully in {process_time} seconds.")
    else:
        print("Process failed.")
