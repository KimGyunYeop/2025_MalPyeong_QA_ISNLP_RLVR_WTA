import json
import argparse
import os
import pandas as pd

from eval_fn import expanded_evaluation_korean_contest_culture_QA

argparser = argparse.ArgumentParser(description="Calculate expanded measurement from JSON file.")
argparser.add_argument("--dev_file_path", type=str, default="data/QA/korean_culture_qa_V1.0_dev+.json", help="Path to the JSON file containing the development data.")
# argparser.add_argument("--target_folder", type=str, default="results", help="split by comma, folder to save the results.")
argparser.add_argument("--target_folder", type=str, default="find_models", help="split by comma, folder to save the results.")

args = argparser.parse_args()

with open(args.dev_file_path, 'r', encoding='utf-8') as f:
    dev_gold_data = json.load(f)
    
files = []
for dirpath, dirnames, filenames in os.walk('.'):   # '.'부터 재귀
    for name in filenames:                          # 파일 이름만
        if "whole" in name and ".json" in name:  # 'whole'와 '.json'이 모두 포함된 파일
            files.append(os.path.join(dirpath, name))   # 상대 경로

print(f"Found {len(files)} files matching the criteria.")
            
results = []
target_folders = args.target_folder.split(',')
print(f"Target folders: {target_folders}")
for target_folder in target_folders:
    for file_path in files:
        if f"{target_folder}/" in file_path:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                whole_data = json.load(f)
            
            if len(whole_data) != len(dev_gold_data):
                print(f"Warning: Length mismatch between whole data and dev gold data in {file_path}.")
                continue

            expanded_measurement = expanded_evaluation_korean_contest_culture_QA(dev_gold_data, whole_data)
            print(f"Expanded measurement for {file_path}: {expanded_measurement}")
            expanded_measurement['file_path'] = file_path
            
            results.append(expanded_measurement)
            
            # Save the results to a new JSON file
            result_file_path = file_path.replace('.json', '_expanded_measurement.json')
            with open(result_file_path, 'w', encoding='utf-8') as result_file:
                json.dump(expanded_measurement, result_file, ensure_ascii=False, indent=4)

# Save all results to a single CSV file
results_df = pd.DataFrame(results)
results_csv_path = 'expanded_measurement_results.csv'
results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
print(f"All results saved to {results_csv_path}.")
