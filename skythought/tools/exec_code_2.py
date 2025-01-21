from bespokelabs import curator
from datasets import load_dataset, Dataset
import json
from util.taco.testing_util import run_test as taco_run_test
import copy
import multiprocessing
from multiprocessing import Manager, Pool
import numpy as np
import re
import math
import os
from tqdm import tqdm
from functools import partial
import timeout_decorator


@timeout_decorator.timeout(10)
def run_test_with_timeout(problem, generation):
    try:
        result = taco_run_test(problem, test=generation, debug=False)
        return bool(result and np.all(result))
    except Exception as e:
        print(f"Error in run_test: {e}")
        return False

def check_correctness(problem, generation):
    try:
        return run_test_with_timeout(problem, generation)
    except timeout_decorator.TimeoutError:
        print("Test execution timed out")
        return False
    except Exception as e:
        print(f"Error in check_correctness: {e}")
        return False

def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches

def process_single_row(row):
    """Process a single row of the dataset"""
    try:
        code_filter_result = has_code(row.get('deepseek_solution', ''))
        
        if len(code_filter_result) == 0:
            row["correctness"] = False
            row["reason"] = "Does not contain code component."
        else:    
            last_code = code_filter_result[-1]
            curr_res = check_correctness(row, last_code)
            if curr_res:
                row["correctness"] = True
                row["reason"] = ""
            else:
                row["correctness"] = False
                row["reason"] = "Code is incorrect."
        
        return row        
    
    except Exception as e:
        row['correctness'] = False
        row['reason'] = f"Processing error: {str(e)}"
        return row

def process_dataset_parallel(df, num_cpus=None, batch_size=1024):
    """Process the dataset in parallel using multiple CPUs"""
    if num_cpus is None:
        num_cpus = max(1, multiprocessing.cpu_count() - 1)
    
    data = df.to_list()
    total_rows = len(data)
    print(f"Processing {total_rows} rows using {num_cpus} CPUs...")

    all_results = []
    for i in range(0, total_rows, batch_size):
        batch = data[i:i+batch_size]
        with Pool(processes=num_cpus) as pool:
            batch_results = list(tqdm(
                pool.map(process_single_row, batch),
                total=len(batch),
                desc=f"Processing batch {i//batch_size + 1}"
            ))
        
        all_results.extend(batch_results)
        
        # Calculate and print statistics for this batch
        batch_correct = sum(1 for r in batch_results if r.get('correctness', False))
        print(f"\nBatch {i//batch_size + 1} Results:")
        print(f"Processed examples: {len(all_results)}/{total_rows}")
        print(f"Correct in this batch: {batch_correct}/{len(batch_results)} ({batch_correct/len(batch_results)*100:.2f}%)")
        print(f"Total correct so far: {sum(1 for r in all_results if r.get('correctness', False))}/{len(all_results)}\n")
    
    return Dataset.from_list(all_results)


if __name__ == '__main__':
    # Set the maximum number of open files limit
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset("bespokelabs/sky-t1-taco-test-unfiltered")

    # Process dataset in parallel
    processed_df = process_dataset_parallel(df['train'])
    

    # processed_df.save_to_disk('final_processed_dataset')
    processed_df.push_to_hub('bespokelabs/sky-t1-taco-train-rejection-sampled-shreyas')
    print("Processing complete! Final dataset saved.")