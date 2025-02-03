import logging

from core.services.predictor import PhonePricePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    market_posts_path = "../input_files/db_data/market_posts.csv"
    phone_main_path = "../input_files/db_data/phone_main.csv"
    phone_models_path = "../input_files/db_data/phone_models.csv"
    phone_models2_path = "../input_files/db_data/phone_models2.csv"

    result_csv_path = "../output_files/output_phone.csv"

    predictor = PhonePricePredictor(market_posts_path, result_csv_path)
    process_time = predictor.run()

    print(f"Process completed successfully in {process_time} seconds.")
