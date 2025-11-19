import argparse
import base64
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock, Thread
from typing import Dict, List, Union, Optional

import pandas as pd
import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

"""
Global variables and configurations
"""
dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(dir, 'config.yaml')
if not os.path.exists(config_path):
    default_config = {'model_list': [{'model': 'your-model-name', 'base_url': 'your-api-base-url', 'api_key': 'your-api-key'}]}
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f)
    logger.warning(f"Config file not found. A default config.yaml has been created. Please edit it.")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

clients = {}
if 'model_list' in config and config['model_list']:
    for model_config in config['model_list']:
        model, base_url, api_key = model_config.get('model'), model_config.get('base_url'), model_config.get('api_key')
        if model and base_url and api_key:
            clients[model] = OpenAI(base_url=base_url, api_key=api_key)
else:
    logger.error("Model configuration is missing or empty in config.yaml.")
    exit(1)

# NOTE: Save the new, modified prompt into 'prompt_en.txt' (or another language file if needed)
languages = ['en'] 
prompts = {}
for l in languages:
    prompt_file_path = os.path.join(dir, f'prompt_{l}_cot.txt')
    if not os.path.exists(prompt_file_path):
        logger.error(f"Prompt file not found at: {prompt_file_path}")
        exit(1)
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompts[l] = f.read()

write_lock = Lock()

"""
Functions
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the reasoning (CoT) of a model's output.")
    parser.add_argument('--input', type=str, required=True, help='Path to the QUESTIONS file (.parquet or .jsonl).')
    # NEW: Argument for the file containing model answers to be evaluated
    parser.add_argument('--answers-input', type=str, required=True, help="Path to the model's ANSWERS file (.jsonl).")
    parser.add_argument('--output', type=str, required=True, help='Path to the evaluation output file.')
    parser.add_argument('--model', type=str, required=True, help='Name of the EVALUATOR model.')
    parser.add_argument('--reasoning-effort', type=str, default=None, choices=['low', 'medium', 'high'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--language', type=str, required=True, choices=languages)
    parser.add_argument('--pass-k', type=int, default=1)
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--n-procs', type=int, default=1)
    parser.add_argument('--n-threads', type=int, default=1)
    return parser.parse_args()

# NEW: Function to load both files and merge them by UUID
def load_and_merge_data(questions_path: str, answers_path: str, language: str) -> pd.DataFrame:
    """
    Loads questions and answers, then merges them based on UUID.
    The final length will match the questions file.
    If multiple answers exist for a UUID, only the first one is used.
    """
    logger.info(f"Loading questions from {questions_path}")
    if questions_path.endswith('.parquet'):
        questions_df = pd.read_parquet(questions_path)
    elif questions_path.endswith('.jsonl'):
        questions_df = pd.read_json(questions_path, lines=True)
    else:
        raise ValueError("Unsupported file format for questions. Use .parquet or .jsonl.")
    
    logger.info(f"Loading answers from {answers_path}")
    if not os.path.exists(answers_path):
        logger.error(f"Answers file not found: {answers_path}")
        exit(1)
    answers_df = pd.read_json(answers_path, lines=True)
    
    answer_col_name = f'model_answer_{language}'
    if 'llm_output' not in answers_df.columns:
        logger.error(f"'llm_output' column not found in {answers_path}")
        exit(1)
    
    answers_df_subset = answers_df[['uuid', 'llm_output']].rename(columns={'llm_output': answer_col_name})
    
    # MODIFICATION 1: Drop duplicate UUIDs from the answers, keeping only the first occurrence.
    # This ensures that each UUID from the questions file can only match one answer at most.
    logger.info(f"Original answers count: {len(answers_df_subset)}. Deduplicating by 'uuid'...")
    answers_df_unique = answers_df_subset.drop_duplicates(subset='uuid', keep='first')
    logger.info(f"Deduplicated answers count: {len(answers_df_unique)}.")
    
    logger.info(f"Merging {len(questions_df)} questions with {len(answers_df_unique)} unique answers on 'uuid'.")
    
    # MODIFICATION 2: Change the merge strategy from 'inner' to 'left'.
    # This keeps all rows from the `questions_df` (the left dataframe).
    # If a UUID from questions_df has no match in answers_df_unique, the 'model_answer' column for that row will be NaN.
    merged_df = pd.merge(questions_df, answers_df_unique, on='uuid', how='left')
    
    # MODIFICATION 3 (Optional but recommended): Fill NaN values for unmatched answers with an empty string.
    # This prevents errors in the prompt generation if a value is NaN.
    merged_df[answer_col_name] = merged_df[answer_col_name].fillna('')
    
    logger.info(f"Successfully merged. Final dataset has {len(merged_df)} entries, matching the questions file.")
    
    # Add a check to see how many answers were successfully matched.
    matched_count = merged_df[answer_col_name].astype(bool).sum()
    logger.info(f"Matched {matched_count} out of {len(questions_df)} questions with an answer.")

    return merged_df


def check_incomplete(input_df: pd.DataFrame, output_df: pd.DataFrame, pass_k: int) -> pd.DataFrame:
    # This function remains unchanged.
    if output_df.empty:
        incomplete_df = input_df.loc[input_df.index.repeat(pass_k)]
    else:
        # Group by UUID and the model that performed the evaluation
        completed_counts = output_df[output_df['status'] == True].groupby('uuid').size()
        remaining_counts = input_df['uuid'].map(lambda x: pass_k - completed_counts.get(x, 0))
        mask = remaining_counts > 0
        incomplete_df = input_df[mask].loc[input_df[mask].index.repeat(remaining_counts[mask])]
    logger.info(f"Found {len(incomplete_df)} incomplete evaluations.")
    return incomplete_df.reset_index(drop=True)

# The rest of the functions (generate_message, parse_evaluation_response, save_result, handle_request, single_task, worker, process_worker)
# are mostly correct from your provided script. I've included them for completeness.

def generate_message(input_line: pd.Series, language: str) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    prompt = prompts[language]
    question = input_line.get(f'question_{language}', '')
    options_dict = input_line.get(f'options_{language}', {})
    ground_truth = input_line.get(f'explanation_{language}', '')
    # This will now correctly get the model's answer from the merged dataframe
    model_answer = input_line.get(f'model_answer_{language}', '') 
    options = ''
    if isinstance(options_dict, dict):
        for key in sorted(options_dict.keys()):
            if options_dict.get(key) is not None:
                options += f"{key}: {options_dict[key]}\n"
    prompt = prompt.replace('{question}', str(question))
    prompt = prompt.replace('{options}', options)
    prompt = prompt.replace('{ground_truth_analysis}', str(ground_truth))
    prompt = prompt.replace('{model_answer}', str(model_answer))
    # print(prompt)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def parse_evaluation_response(response: str) -> Dict:
    """
    Parse the model's response to extract the evaluation JSON.
    Args:
        response: The response string from the model.
    Returns:
        A dictionary parsed from the JSON output.
    """
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, response)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to find the first JSON-like object if no markdown block is found
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("No valid JSON structure found in the response.")
    
    try:
        parsed = json.loads(json_str)
        # We only need to ensure 'checkpoint_details' exists and is a list.
        if 'checkpoint_details' not in parsed:
            raise ValueError("Parsed JSON is missing the required 'checkpoint_details' key.")
        if not isinstance(parsed['checkpoint_details'], list):
            raise ValueError("'checkpoint_details' must be a list.")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON. Error: {e}")

def save_result(output_path: str, result: dict):
    assert output_path.endswith('.jsonl'), "Output file must be in .jsonl format."
    with write_lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + '\n')

def handle_request(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, timeout: int):
    client = clients[model]
    messages = generate_message(input_line, language)
    parameters = {
        'model': model, 'messages': messages, 'temperature': temperature, 'timeout': timeout,
        'stream': True, 'max_tokens': 16384, 'stream_options': {"include_usage": True},
    }
    if reasoning_effort: parameters['reasoning_effort'] = reasoning_effort
    response = client.chat.completions.create(**parameters)
    content, finish_reason, prompt_usage, completion_usage = '', None, 0, 0
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content: content += delta.content
            if chunk.choices[0].finish_reason: finish_reason = chunk.choices[0].finish_reason
        if hasattr(chunk, 'usage') and chunk.usage:
            prompt_usage, completion_usage = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
    if finish_reason != 'stop':
        logger.warning(f"Model finished with reason: {finish_reason} for UUID {input_line.get('uuid')}.")
    return content, finish_reason, prompt_usage, completion_usage

def single_task(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, max_retries: int, timeout: int) -> Dict:
    """
    Handle a single evaluation task, parse the response, and calculate the score.
    """
    retries_count, time_wait = 0, 5
    while retries_count < max_retries:
        try:
            content, finish_reason, prompt_usage, completion_usage = handle_request(
                input_line, model, reasoning_effort, temperature, language, timeout)
            
            parsed_evaluation = parse_evaluation_response(content)
            
            # --- Score Calculation Logic ---
            details = parsed_evaluation.get('checkpoint_details', [])
            
            # Count the total number of checkpoints
            total_checkpoints = len(details)
            
            # Count how many checkpoints have "is_matched": true
            matched_checkpoints = 0
            if total_checkpoints > 0:
                matched_checkpoints = sum(1 for item in details if isinstance(item, dict) and item.get('is_matched') is True)

            return {
                'uuid': input_line['uuid'],
                # NEW: Save the calculated counts
                'matched_checkpoints': matched_checkpoints,
                'total_checkpoints': total_checkpoints,
                # The detailed breakdown is still valuable to save
                'evaluation_details': details,
                'llm_output': content, # The raw JSON output is also useful
                'finish_reason': finish_reason,
                'prompt_usage': prompt_usage,
                'completion_usage': completion_usage,
                'model': model, # The evaluator model
                'reasoning_effort': reasoning_effort,
                'temperature': temperature,
                'language': language,
                'status': True,
            }
        except Exception as e:
            retries_count += 1
            logger.error(f"Error on UUID {input_line['uuid']} with model {model}: {e}. Retrying {retries_count}/{max_retries}...")
            if retries_count >= max_retries:
                return {
                    'uuid': input_line['uuid'], 'model': model, 'reasoning_effort': reasoning_effort,
                    'temperature': temperature, 'language': language, 'status': False, 'error': str(e)
                }
            time.sleep(time_wait)

def worker(task_queue, args, process_id, thread_id, progress_counter, progress_lock):
    output_path, model, reasoning_effort, temp, lang, max_retries, timeout = args
    while True:
        try:
            task_data = task_queue.get(timeout=1)
            if task_data is None: break
            result = single_task(pd.Series(task_data), model, reasoning_effort, temp, lang, max_retries, timeout)
            save_result(output_path, result)
            with progress_lock: progress_counter.value += 1
        except Exception as e:
            if "Empty" in str(type(e)): continue
            logger.error(f"Error in worker {process_id}-{thread_id}: {e}")
            break

def process_worker(task_queue, args, n_threads, process_id, progress_counter, progress_lock):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker, task_queue, args, process_id, i, progress_counter, progress_lock) for i in range(n_threads)]
        for future in futures: future.result()

def main():
    manager = mp.Manager()
    task_queue = manager.Queue()
    args = parse_args()
    logger.info(f"Starting CoT evaluation with evaluator model: {args.model}")
    
    # MODIFIED: Use the new data loading and merging function
    merged_df = load_and_merge_data(args.input, args.answers_input, args.language)
    
    output_df = pd.read_json(args.output, lines=True) if os.path.exists(args.output) else pd.DataFrame()
    incomplete_df = check_incomplete(merged_df, output_df, args.pass_k)

    if incomplete_df.empty:
        logger.info("No incomplete tasks found for the given inputs. Exiting.")
        return
    
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()
    
    with tqdm(total=len(incomplete_df), desc="Processing evaluations") as pbar:
        def update_progress():
            last_val = 0
            while progress_counter.value < len(incomplete_df):
                if progress_counter.value > last_val:
                    pbar.update(progress_counter.value - last_val)
                    last_val = progress_counter.value
                time.sleep(0.1)
            pbar.update(len(incomplete_df) - last_val)

        progress_thread = Thread(target=update_progress)
        progress_thread.start()

        for _, row in incomplete_df.iterrows(): task_queue.put(row.to_dict())
        for _ in range(args.n_procs * args.n_threads): task_queue.put(None)
        
        worker_args = (args.output, args.model, args.reasoning_effort, args.temperature, args.language,
                       args.max_retries, args.timeout)
        
        if args.n_procs == 1:
            process_worker(task_queue, worker_args, args.n_threads, 0, progress_counter, progress_lock)
        else:
            with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
                futures = [executor.submit(process_worker, task_queue, worker_args, args.n_threads, i, progress_counter, progress_lock) for i in range(args.n_procs)]
                for future in futures: future.result()

        progress_thread.join()
        
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()