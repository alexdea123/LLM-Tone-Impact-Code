# # borrowed and extended from
# # https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

# import os
# import sys
# import gc
# import psutil

# sys.set_int_max_str_digits(50000)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import json
# import multiprocessing
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor, as_completed


# import numpy as np
# from tqdm import tqdm

# from lcb_runner.evaluation.testing_util import run_test
# from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results


# def _temp_run(sample, generation, debug, result, metadata_list, timeout):
#     res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
#     result.append(res)
#     metadata_list.append(metadata)


# def check_correctness(sample, generation, timeout, debug=True):
#     """Check correctness of code generation with a global timeout.
#     The global timeout is to catch some extreme/rare cases not handled by the timeouts
#     inside `run_test`"""

#     # Calculate a reasonable but limited timeout
#     try:
#         in_outs = json.loads(sample["input_output"])
#         # Hard cap the timeout to prevent resource exhaustion
#         max_timeout = min((timeout + 1) * len(in_outs["inputs"]) + 5, 30)  # Maximum 15 seconds
#     except Exception:
#         max_timeout = min(timeout + 5, 15)  # Default fallback

#     # Create a separate process with limited resources
#     manager = multiprocessing.Manager()
#     result = manager.list()
#     metadata_list = manager.list()
    
#     try:
#         p = multiprocessing.Process(
#             target=_temp_run,
#             args=(sample, generation, debug, result, metadata_list, timeout),
#         )
#         p.daemon = True  # Make sure the process dies if the parent dies
#         p.start()
#         p.join(timeout=max_timeout)
        
#         if p.is_alive():
#             p.terminate()
#             p.join(1)  # Give it 1 second to terminate
#             if p.is_alive():
#                 p.kill()  # Force kill if still alive
#                 gc.collect()
                
#             if debug:
#                 print(f"Process killed after {max_timeout}s timeout")
                
#             # Create failure result - NOTE: Fixed nesting issue here
#             try:
#                 in_outs = json.loads(sample["input_output"])
#                 result_val = [-1 for _ in range(len(in_outs["inputs"]))]
#             except:
#                 result_val = [-1]
                
#             metadata_val = {
#                 "error": "Timeout",
#                 "error_code": -5,
#                 "error_message": f"Execution timed out after {max_timeout}s"
#             }
#         else:
#             # Process completed normally
#             result_val = result[0] if result else [-1]
#             metadata_val = metadata_list[0] if metadata_list else {"error": "No result"}
    
#     except Exception as e:
#         print(f"Error in check_correctness: {e}")
#         result_val = [-1]
#         metadata_val = {
#             "error": str(e),
#             "error_code": -5,
#             "error_message": "Exception in check_correctness"
#         }
    
#     finally:
#         # Clean up manager resources
#         manager.shutdown()
#         gc.collect()
        
#     return result_val, metadata_val


# def evaluate_generations_by_problem(args):
#     problem_generations: list[str] = args[0]
#     sample = args[1]
#     debug: bool = args[2]
#     timeout: int = args[3]

#     res = []
#     metadata = []
#     for o_idx, o in enumerate(problem_generations):
#         curr_res = [-2]
#         try:
#             curr_res, curr_metadata = check_correctness(
#                 sample, o, timeout=timeout, debug=debug
#             )
#             if debug:
#                 print(f"\nSuccessful compilation of task {o_idx}!")
#             fixed = []
#             for e in curr_res:
#                 if isinstance(e, np.ndarray):
#                     e = e.item(0)
#                 if isinstance(e, np.bool_):
#                     e = bool(e)
#                 fixed.append(e)
#             curr_res = fixed
#             if not np.all(curr_res):
#                 if debug:
#                     print(f"Results were not True for all test cases {curr_res=}\n")
#         except Exception as e:
#             if debug:
#                 print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
#             # break
#             curr_metadata = {
#                 "error": repr(e),
#                 "error_code": -5,
#                 "error_message": "TestRunnerError",
#             }
#         finally:
#             assert isinstance(curr_res, list), curr_res
#             assert isinstance(curr_metadata, dict), curr_metadata
#             res.append(curr_res)
#             metadata.append(curr_metadata)
#     if debug:
#         for i, r in enumerate(problem_generations):
#             print("Sample\n")
#             print(r)
#             print("\n")
#             print("Result\n")
#             print(res[i])
#             print("*" * 30 + "\n\n")
#     return res, metadata


# def evaluate_generations(
#     samples_list: list,
#     generations_list: list[list[str]],
#     debug: bool = False,
#     num_process_evaluate: int = 16,
#     timeout=6,
# ):
#     total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
#     # Be conservative with process count based on available memory
#     safe_process_count = max(1, min(num_process_evaluate, int(total_memory_gb / 2)))
    
#     if safe_process_count < num_process_evaluate:
#         print(f"Limiting process count to {safe_process_count} based on available memory")
#         num_process_evaluate = safe_process_count

#     # Create inputs for each evaluation task
#     inputs = [
#         [(generations_list[index], samples_list[index], debug, timeout), index]
#         for index in range(len(generations_list))
#     ]

#     results = {}
#     metadata = {}

#     with tqdm(total=len(inputs)) as pbar:
#         # Use a context manager to ensure resources are released
#         with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
#             # Submit all jobs
#             futures = {
#                 executor.submit(evaluate_generations_by_problem, arg): index
#                 for arg, index in inputs
#             }

#             # Process results as they complete
#             for future in as_completed(futures):
#                 try:
#                     index = futures[future]
#                     result, meta = future.result()
#                     results[index], metadata[index] = result, meta
#                     pbar.update(1)
#                 except Exception as e:
#                     print(f"Error processing job: {e}")
                
#                 # Force some cleanup after each completion
#                 gc.collect()

#     # Verify all results are collected
#     assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)}"
    
#     return results, metadata

# def codegen_metrics(
#     samples_list,
#     generations_list,
#     k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
#     num_process_evaluate=16,
#     timeout=6,
#     debug=False,
#     batch_size=100,  # Process in smaller batches
# ):
#     import gc  # Add garbage collection

#     # Get total size for progress tracking
#     total_samples = len(samples_list)
#     total_evaluated = 0
    
#     # Prepare result containers
#     all_results = defaultdict(list)
#     all_metadatas = defaultdict(list)
    
#     # Process in batches to limit memory usage
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         batch_size_actual = batch_end - batch_start
        
#         print(f"Processing batch {batch_start}-{batch_end} of {total_samples}...")
        
#         # Process only the current batch
#         batch_samples = samples_list[batch_start:batch_end]
#         batch_generations = generations_list[batch_start:batch_end]
        
#         # Linearize the batch
#         samples_linear = []
#         generations_linear = []
#         remap_index = []
        
#         for idx, (sample, generation_list) in enumerate(zip(batch_samples, batch_generations)):
#             assert isinstance(generation_list, list), generation_list
#             for generation in generation_list:
#                 assert isinstance(generation, str), generation
#                 samples_linear.append(sample)
#                 generations_linear.append([generation])
#                 remap_index.append(batch_start + idx)
        
#         # Adjust num_process_evaluate based on batch size
#         effective_processes = min(num_process_evaluate, max(1, batch_size_actual // 4))
        
#         # Run evaluation on the batch
#         results_linear, metadatas_linear = evaluate_generations(
#             samples_linear,
#             generations_linear,
#             debug=debug,
#             num_process_evaluate=effective_processes, 
#             timeout=timeout,
#         )
        
#         # Process results for this batch
#         for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
#             all_results[remap_index[idx]].append(sub_results[0])
        
#         for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
#             all_metadatas[remap_index[idx]].append(sub_metadatas[0])
        
#         # Explicitly release memory
#         del samples_linear, generations_linear, remap_index
#         del results_linear, metadatas_linear
#         gc.collect()
        
#         total_evaluated += batch_size_actual
#         print(f"Processed {total_evaluated}/{total_samples} samples")
    
#     # Compute metrics once all batches are processed
#     metrics = compute_metrics_from_results(all_results, k_list=k_list)
    
#     # Process final metadata
#     final_metadata = []
#     for key in sorted(list(all_metadatas.keys())):
#         metadata_entry = all_metadatas[key]
#         if not isinstance(metadata_entry, list):
#             metadata_entry = [json.dumps(metadata_entry)]
#         else:
#             metadata_entry = [json.dumps(x) for x in metadata_entry]
        
#         # Verify lengths match
#         expected_length = len(generations_list[key % len(generations_list)])
#         if len(metadata_entry) != expected_length:
#             print(f"Warning: Metadata length mismatch for key {key}: {len(metadata_entry)} vs {expected_length}")
#             # Pad or truncate to match expected length
#             if len(metadata_entry) < expected_length:
#                 metadata_entry.extend([json.dumps({})] * (expected_length - len(metadata_entry)))
#             else:
#                 metadata_entry = metadata_entry[:expected_length]
                
#         final_metadata.append(metadata_entry)
    
#     return [metrics, all_results, final_metadata]


# if __name__ == "__main__":
#     # print(
#     #     check_correctness(
#     #         {
#     #             "input_output": json.dumps(
#     #                 {
#     #                     "inputs": [
#     #                         json.dumps([1] * 100000)
#     #                         + "\n"
#     #                         + json.dumps([100000, -100000] * (100000 // 2))
#     #                     ],
#     #                     "outputs": [json.dumps([100000, 0] * (100000 // 2))],
#     #                     "fn_name": "mostFrequentIDs",
#     #                 }
#     #             )
#     #         },
#     #         "class Solution:\n    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:\n        from collections import defaultdict\n        \n        # Count of each ID\n        count = defaultdict(int)\n        # How many IDs exist for a given frequency\n        freq_of_count = defaultdict(int)\n        \n        max_freq = 0\n        ans = []\n        \n        for i in range(len(nums)):\n            x = nums[i]\n            change = freq[i]\n            \n            old_freq = count[x]\n            new_freq = old_freq + change\n            \n            # If there was an old frequency, decrease its usage\n            if old_freq > 0:\n                freq_of_count[old_freq] -= 1\n                if freq_of_count[old_freq] == 0:\n                    del freq_of_count[old_freq]\n            \n            # Update with the new frequency\n            count[x] = new_freq\n            freq_of_count[new_freq] += 1\n            \n            # Update max_freq if needed\n            if new_freq > max_freq:\n                max_freq = new_freq\n            \n            # If the collection at max_freq is empty, reduce max_freq until we find a non-empty bin\n            while max_freq > 0 and max_freq not in freq_of_count:\n                max_freq -= 1\n            \n            # If the collection is empty, max_freq will be 0\n            ans.append(max_freq)\n        \n        return ans",
#     #         6,
#     #         debug=True,
#     #     )
#     # )

#     print(
#         check_correctness(
#             {
#                 "input_output": json.dumps(
#                     {
#                         "inputs": ")))))",
#                         "outputs": "0",
#                     },
#                 )
#             },
#             "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
#             6,
#             debug=True,
#         )
#     )
import os
import sys
import gc
# import psutil

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results


# ----- Modified _temp_run: return results directly instead of writing into a Manager list -----
def _temp_run(sample, generation, debug, timeout):
    # Directly return the (result, metadata) tuple from run_test.
    # (Assume run_test is imported from lcb_runner.evaluation.testing_util)
    return run_test(sample, test=generation, debug=debug, timeout=timeout)

# ----- Modified check_correctness: use ProcessPoolExecutor without Manager -----
def check_correctness(sample, generation, timeout, debug=True):
    """
    Check correctness with a global timeout using ProcessPoolExecutor.
    Eliminates Manager overhead by returning the result directly.
    """
    # Compute a capped timeout based on the sample inputs.
    try:
        in_outs = json.loads(sample["input_output"])
        max_timeout = min((timeout + 1) * len(in_outs["inputs"]) + 5, 30)
    except Exception:
        max_timeout = min(timeout + 5, 15)

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_temp_run, sample, generation, debug, timeout)
        try:
            result, metadata = future.result(timeout=max_timeout)
        except Exception as e:
            future.cancel()
            try:
                in_outs = json.loads(sample["input_output"])
                result = [-1 for _ in range(len(in_outs["inputs"]))]
            except Exception:
                result = [-1]
            metadata = {
                "error": str(e),
                "error_code": -5,
                "error_message": f"Execution timed out after {max_timeout}s"
            }
    gc.collect()
    return result, metadata

# ----- Modified evaluate_generations_by_problem: streamlined without extra Manager calls -----
def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        try:
            curr_res, curr_metadata = check_correctness(sample, o, timeout=timeout, debug=debug)
            # Convert any numpy types to native Python types.
            fixed = [
                e.item(0) if isinstance(e, np.ndarray) else bool(e) if isinstance(e, np.bool_) else e
                for e in curr_res
            ]
            curr_res = fixed
        except Exception as e:
            curr_res = [-1]
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError"
            }
        res.append(curr_res)
        metadata.append(curr_metadata)
        if debug:
            print(f"Task {o_idx} result: {curr_res}")
        gc.collect()
    return res, metadata

# ----- Modified evaluate_generations: use futures directly, remove extraneous gc calls -----
def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    # Limit process count based on available memory.
    # total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    total_memory_gb = 1
    safe_process_count = max(1, min(num_process_evaluate, int(total_memory_gb / 2)))
    if safe_process_count < num_process_evaluate:
        print(f"Limiting process count to {safe_process_count} based on available memory")
        num_process_evaluate = safe_process_count

    # Build list of evaluation tasks.
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]
    results = {}
    metadata = {}

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }
            for future in as_completed(futures):
                try:
                    index = futures[future]
                    result_val, meta_val = future.result()
                    results[index] = result_val
                    metadata[index] = meta_val
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing job: {e}")
                gc.collect()

    assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)}"
    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
    batch_size=100,  # Process in smaller batches
):
    # Get total size for progress tracking
    total_samples = len(samples_list)
    total_evaluated = 0
    
    # Prepare result containers
    all_results = defaultdict(list)
    all_metadatas = defaultdict(list)
    
    # Process in batches to limit memory usage
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_size_actual = batch_end - batch_start
        
        print(f"Processing batch {batch_start}-{batch_end} of {total_samples}...")
        
        # Process only the current batch
        batch_samples = samples_list[batch_start:batch_end]
        batch_generations = generations_list[batch_start:batch_end]
        
        # Linearize the batch
        samples_linear = []
        generations_linear = []
        remap_index = []
        
        for idx, (sample, generation_list) in enumerate(zip(batch_samples, batch_generations)):
            assert isinstance(generation_list, list), generation_list
            for generation in generation_list:
                assert isinstance(generation, str), generation
                samples_linear.append(sample)
                generations_linear.append([generation])
                remap_index.append(batch_start + idx)
        
        # Adjust num_process_evaluate based on batch size
        effective_processes = min(num_process_evaluate, max(1, batch_size_actual // 4))
        
        # Run evaluation on the batch
        results_linear, metadatas_linear = evaluate_generations(
            samples_linear,
            generations_linear,
            debug=debug,
            num_process_evaluate=effective_processes, 
            timeout=timeout,
        )
        
        # Process results for this batch
        for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
            all_results[remap_index[idx]].append(sub_results[0])
        
        for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
            all_metadatas[remap_index[idx]].append(sub_metadatas[0])
        
        # Explicitly release memory
        del samples_linear, generations_linear, remap_index
        del results_linear, metadatas_linear
        gc.collect()
        
        total_evaluated += batch_size_actual
        print(f"Processed {total_evaluated}/{total_samples} samples")
    
    # Compute metrics once all batches are processed
    metrics = compute_metrics_from_results(all_results, k_list=k_list)
    
    # Process final metadata
    final_metadata = []
    for key in sorted(list(all_metadatas.keys())):
        metadata_entry = all_metadatas[key]
        if not isinstance(metadata_entry, list):
            metadata_entry = [json.dumps(metadata_entry)]
        else:
            metadata_entry = [json.dumps(x) for x in metadata_entry]
        
        # Verify lengths match
        expected_length = len(generations_list[key % len(generations_list)])
        if len(metadata_entry) != expected_length:
            print(f"Warning: Metadata length mismatch for key {key}: {len(metadata_entry)} vs {expected_length}")
            # Pad or truncate to match expected length
            if len(metadata_entry) < expected_length:
                metadata_entry.extend([json.dumps({})] * (expected_length - len(metadata_entry)))
            else:
                metadata_entry = metadata_entry[:expected_length]
                
        final_metadata.append(metadata_entry)
    
    return [metrics, all_results, final_metadata]
