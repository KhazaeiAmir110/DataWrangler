# Data Processing and Cleaning Script

## Overview
This script processes and cleans mobile phone market data to generate meaningful statistics. It filters, merges, and analyzes CSV datasets to compute pricing metrics like mean price, standard deviation, and error percentage. The final output is saved as a CSV file.

## Features
- Reads and validates input files
- Filters out invalid and noisy data
- Computes weighted mean prices and standard deviations
- Determines lower and upper price bounds based on statistical analysis
- Calculates error percentages for price estimation
- Outputs the cleaned and structured dataset to a CSV file

## Prerequisites
Make sure you have Python installed along with the required libraries:
pip install numpy pandas argparse

## Input Files
The script expects the following CSV files in the input directory:
- market_posts.csv: Contains market listings for mobile phones
- phone_main2.csv: Contains primary phone model details
- reference_phone_prices.csv: Reference dataset for phone pricing
- phone_models.csv: Detailed phone model specifications

## Running the Script
The script requires two arguments:
1. Input directory (path containing the CSV files)
2. Output directory (path where the processed CSV file will be saved)

To execute the script, run:
python main.py <input_directory> <output_directory>
Example:
python main.py /path/to/input /path/to/output

## Output File
The script generates an output CSV file in the specified output directory:
- `output.csv`
- Contains relevant pricing and error metrics for each phone model

## Key Calculations
1. Weighted Mean Price: Calculates a time-weighted mean price for each phone model.
2. Price Bounds:
   - Lower Bound: mean_price - 0.8 * std_dev
   - Upper Bound: mean_price + 1.2 * std_dev
3. Error Percentage:
   - Lower Error: abs((reference_lower_bound / lower_bound * 100) - 100)
   - Upper Error: abs((reference_upper_bound / upper_bound * 100) - 100)
   - Average Error: (lower_error + upper_error) / 2

## Error Handling
- Checks for missing files before execution.
- Ensures required columns exist before merging datasets.
- Handles NaN values in key numerical columns.
- Creates the output directory if it does not exist.