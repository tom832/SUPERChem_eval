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
        model = model_config.get('model')
        base_url = model_config.get('base_url')
        api_key = model_config.get('api_key')
        if model and base_url and api_key:
            clients[model] = OpenAI(base_url=base_url, api_key=api_key)
else:
    logger.error("Model configuration is missing or empty in config.yaml.")
    exit(1)


languages = ['en'] 
prompts = {}
for l in languages:
    prompt_file_path = os.path.join(dir, f'prompt_{l}_ckpt.txt')
    if not os.path.exists(prompt_file_path):
        logger.error(f"Prompt file not found at: {prompt_file_path}")
        # If the prompt file does not exist, create an empty one and prompt the user
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write("Please paste your tagging prompt here. Use {question}, {options}, and {explanation} as placeholders.")
        logger.warning(f"An empty prompt file has been created. Please fill it with your prompt content.")
        exit(1)
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompts[l] = f.read()

write_lock = Lock()

"""
Functions
"""
def parse_args() -> argparse.Namespace:
    """
    - input: str, required, the path to the result file.
    - output: str, required, the path to the output file.
    - model: str, required, the name of the model to evaluate.
    - reasoning-effort: str, default None, the reasoning effort for the model's output (low, medium, high).
    - temperature: float, optional, the temperature for the model's output.
    - language: str, required, 'zh' or 'en', the language of the model's input.
    - multimodal: bool, required, whether the input is multimodal or not.
    - pass-k: int, default 1, the number of passes to evaluate.
    - max-retries: int, default 5, the maximum number of retries for a request.
    - timeout: int, default 300, the timeout for each request in seconds.
    - n-procs: int, default 1, the number of processes to use for evaluation.
    - n-threads: int, default 1, the number of threads to use for evaluation.
    - log-level: str, optional, the logging level (DEBUG, INFO, WARNING, ERROR).
    """
    parser = argparse.ArgumentParser(description="Evaluate the results of a model.")
    parser.add_argument('--input', type=str, required=True, help='Path to the result file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to evaluate.')
    parser.add_argument('--reasoning-effort', type=str, default=None, choices=['low', 'medium', 'high'], help='Reasoning effort for the model\'s output.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the model\'s output.')
    parser.add_argument('--language', type=str, required=True, choices=languages, help='Language of the model\'s input.')
    parser.add_argument('--multimodal', type=bool, required=True, help='Whether the input is multimodal or not.')
    parser.add_argument('--pass-k', type=int, default=1, help='Number of passes to evaluate.')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries for a request.')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout for each request in seconds.')
    parser.add_argument('--n-procs', type=int, default=1, help='Number of processes to use for evaluation.')
    parser.add_argument('--n-threads', type=int, default=1, help='Number of threads to use for evaluation.')
    return parser.parse_args()

def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the input and output data from the specified path. Support .parquet and .jsonl format.
    """
    logger.info(f"Loading data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        exit(1)
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.jsonl'):
        df = pd.read_json(input_path, lines=True)
    else:
        logger.error(f"Unsupported file format: {input_path}. Only .parquet and .jsonl are supported.")
        exit(1)
    return df

def check_incomplete(input_df: pd.DataFrame, output_df: pd.DataFrame, pass_k: int) -> pd.DataFrame:
    """
    Check for incomplete evaluations in the output DataFrame.
    Args:
        input_df: The input DataFrame containing the questions.
        output_df: The output DataFrame containing the model's responses.
        pass_k: The number of passes to evaluate.
    Returns:
        A DataFrame containing the incomplete evaluations.
    """
    # Check if output_df is empty
    if output_df.empty:
        # Repeat all input rows
        incomplete_df = input_df.loc[input_df.index.repeat(pass_k)]
    else:
        # Count successful responses for each uuid
        completed_counts = output_df[output_df['status'] == True].groupby('uuid').size()
        # Calculate remaining needed for each uuid in input_df
        remaining_counts = input_df['uuid'].map(lambda x: pass_k - completed_counts.get(x, 0))
        # Filter and repeat rows based on remaining counts
        mask = remaining_counts > 0
        incomplete_df = input_df[mask].loc[input_df[mask].index.repeat(remaining_counts[mask])]
    logger.info(f"Found {len(incomplete_df)} incomplete evaluations.")

    return incomplete_df.reset_index(drop=True)
# MODIFICATION 1: Major changes to the generate_message function to adapt to the new prompt structure
def generate_message(input_line: pd.Series, language: str, multimodal: bool) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    """
    Generate the message for the model based on the new checkpoint-tagging prompt.
    """
    prompt_template = prompts[language]

    question = input_line.get(f'question_{language}', '')
    options_dict = input_line.get(f'options_{language}', {})
    
    checkpoints_list = input_line.get(f'explanation_{language}', [])

    options_str = ''
    if isinstance(options_dict, dict):
        for key in sorted(options_dict.keys()):
            if options_dict[key] is not None:
                options_str += f"{key}: {options_dict[key]}\n"
    
    # New: Format the checkpoints list into a numbered string
    checkpoints_str = "\n".join([f"{i+1}. {cp}" for i, cp in enumerate(checkpoints_list)])

    # Fill the new prompt template
    prompt = prompt_template.replace('{question}', str(question))
    prompt = prompt.replace('{options}', options_str)
    prompt = prompt.replace('{checkpoints}', checkpoints_str)

    # Multimodal processing logic remains unchanged
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'
    # ... (This part of the multimodal image processing code is the same as your original one and does not need to be modified) ...
    # ... (Code omitted as the logic is unchanged) ...
    
    # Ensure the correct message format is returned
    # Note: The previously unshown multimodal code needs to be kept complete
    # For simplicity, only the plain text logic is shown here
    if not multimodal:
         message = prompt # Simplified example
    else:
        # This should be your complete multimodal message construction logic
        message = [] # Dummy multimodal message
        # ... (Your original complete multimodal message construction code is omitted here, please make sure it is here) ...

    # Assuming your multimodal processing logic can correctly handle the new prompt string
    # (Your original code was built based on the prompt variable, so it should be fine)
    
    # Ensure a list containing a dictionary is returned
    messages = [{'role': 'user', 'content': prompt}] # Simplified plain text return
    # If it is multimodal, `content` will be a list, which your original code has already handled
    return messages


# MODIFICATION 2: Major changes to the parse_tagging_response function to parse the new JSON structure
def parse_tagging_response(response: str) -> Dict[str, List[Dict]]:
    """
    Parse the model's response to extract the checkpoints_analysis list.
    """
    # JSON extraction logic remains unchanged
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, response)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("No valid JSON structure found in the response.")

    try:
        parsed = json.loads(json_str)
        # Core validation logic changed
        if 'checkpoints_analysis' not in parsed:
            raise ValueError("Parsed JSON is missing the 'checkpoints_analysis' key.")
        if not isinstance(parsed['checkpoints_analysis'], list):
            raise ValueError("'checkpoints_analysis' must be a list.")
        
        # (Optional) Stricter validation to check the structure of objects in the list
        for item in parsed['checkpoints_analysis']:
            if not all(k in item for k in ['checkpoint_text', 'knowledge_tags', 'ability_tags']):
                 raise ValueError("An object in 'checkpoints_analysis' is missing required keys.")

        return parsed # Return the entire parsed dictionary directly
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON. Error: {e}")


def save_result(output_path: str, result: dict):
    """
    Save an additional result to the tail of the output .jsonl file.
    """
    assert output_path.endswith('.jsonl'), "Output file must be in .jsonl format."
    with write_lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + '\n')

def handle_request(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, multimodal: bool, timeout: int):
    """
    Send a single request to the model and return the response.
    (MODIFIED to always use streaming for robustness)
    """
    client = clients[model]
    messages = generate_message(input_line, language, multimodal)
    
    # --- Logic is now simplified: always stream ---
    parameters = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'timeout': timeout,
        'stream': True, # Always set to True
        'max_tokens': 16384,
        'stream_options': {"include_usage": True},
    }
    if reasoning_effort in ['low', 'medium', 'high']:
        parameters['reasoning_effort'] = reasoning_effort
    
    response = client.chat.completions.create(**parameters)
    
    # --- Process the stream ---
    reasoning = ''
    content = ''
    finish_reason = None
    prompt_usage = None
    completion_usage = None

    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Collect reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning += delta.reasoning_content
            # Collect main content
            if hasattr(delta, 'content') and delta.content:
                content += delta.content
            # Get the finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        
        # The last chunk of the stream contains the usage data
        if hasattr(chunk, 'usage') and chunk.usage:
            prompt_usage = chunk.usage.prompt_tokens
            completion_usage = chunk.usage.completion_tokens

    # Fallback for models that mix reasoning in content with <think> tags
    if not reasoning and '<think>' in content:
        matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
        if matches:
            reasoning = '\n\n---\n\n'.join(matches)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Check if the model finished correctly
    if finish_reason != 'stop':
        if finish_reason is None:
            logger.warning(f"Model {model} for input {input_line['uuid']} did not return a finish reason. Stream may have been interrupted.")
        else:
            logger.warning(f"Model {model} for input {input_line['uuid']} finished with reason: {finish_reason}. The output might be incomplete.")
            
    # In case the stream is cut off before usage stats are sent
    if prompt_usage is None:
        prompt_usage = 0
    if completion_usage is None:
        completion_usage = 0
            
    return content, reasoning, finish_reason, prompt_usage, completion_usage
# MODIFICATION 3: Major changes to the single_task function to store the new output structure
def single_task(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, multimodal: bool, max_retries: int, timeout: int) -> Dict[str, Union[str, bool, list, dict]]:
    """
    Handle a single checkpoint-tagging task.
    """
    retries_count = 0
    time_wait = 5
    time_multiplier = 0.5
    while retries_count < max_retries:
        try:
            # The content returned by handle_request remains unchanged
            full_response = handle_request(input_line, model, reasoning_effort, temperature, language, multimodal, timeout)
            content, reasoning, finish_reason, prompt_usage, completion_usage = full_response
            
            # Use the new parsing function
            parsed_data = parse_tagging_response(content)

            # Return the new data structure
            return {
                'uuid': input_line['uuid'],
                # Directly store the entire parsed 'checkpoints_analysis' list
                'checkpoints_analysis': parsed_data['checkpoints_analysis'], 
                'llm_output': content,
                'llm_reasoning': reasoning,
                'finish_reason': finish_reason,
                'prompt_usage': prompt_usage,
                'completion_usage': completion_usage,
                'model': model,
                'reasoning_effort': reasoning_effort,
                'temperature': temperature,
                'language': language,
                'multimodal': multimodal,
                'status': True,
            }
        except Exception as e:
            # Exception handling logic remains unchanged
            retries_count += 1
            logger.error(f"Error processing {input_line['uuid']} with model {model}: {e}. Retrying {retries_count}/{max_retries}...")
            if retries_count >= max_retries:
                logger.error(f"Max retries reached for {input_line['uuid']} with model {model}.")
                return {
                    'uuid': input_line['uuid'],
                    # ... (other failure information) ...
                    'status': False,
                    'error': str(e)
                }
            time.sleep(time_wait)
            time_wait += time_multiplier * retries_count

# The rest of the code (worker functions and main) remains unchanged as it's generic
def worker(task_queue, args, process_id, thread_id, progress_counter, progress_lock):
    """Worker function for processing tasks"""
    logger.info(f"Process {process_id}, Thread {thread_id} started")
    output_path, model, reasoning_effort, temperature, language, multimodal, max_retries, timeout = args
    while True:
        try:
            task_data = task_queue.get(timeout=1)
            if task_data is None:  # Poison pill
                break
            # Make sure task_data is a pandas Series
            if isinstance(task_data, dict):
                 input_line = pd.Series(task_data)
            else:
                 input_line = task_data
            result = single_task(input_line, model, reasoning_effort, temperature, language, multimodal, max_retries, timeout)
            save_result(output_path, result)
            with progress_lock:
                progress_counter.value += 1
        except Exception as e:
            if "Empty" in str(type(e)):
                continue
            logger.error(f"Error in process {process_id}, thread {thread_id}: {e}")
            break
    logger.info(f"Process {process_id}, Thread {thread_id} finished")

def process_worker(task_queue, args, n_threads, process_id, progress_counter, progress_lock):
    """Process worker that spawns multiple threads"""
    logger.info(f"Process {process_id} started with {n_threads} threads")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker, task_queue, args, process_id, thread_id, progress_counter, progress_lock) 
                   for thread_id in range(n_threads)]
        for future in futures:
            future.result()
    logger.info(f"Process {process_id} finished")

def main():
    manager = mp.Manager()
    task_queue = manager.Queue()

    args = parse_args()
    logger.info(f"Loading configuration from {dir}")
    logger.info(f"Clients initialized for models: {list(clients.keys())}")
    logger.info(f"Starting evaluation with model: {args.model}, n_procs: {args.n_procs}, n_threads: {args.n_threads}")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"[Prompt]\n\n{prompts[args.language]}")
    
    # Load data
    input_df = load_data(args.input)
    
    # Check if output file exists
    if os.path.exists(args.output):
        try:
            output_df = pd.read_json(args.output, lines=True)
        except ValueError: # Handle empty file case
            output_df = pd.DataFrame()
    else:
        output_df = pd.DataFrame()
    
    # Check incomplete tasks
    incomplete_df = check_incomplete(input_df, output_df, args.pass_k)

    if len(incomplete_df) == 0:
        logger.info("No incomplete tasks found. Exiting.")
        return
    
    # Initialize progress bar
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()

    def update_progress_bar(pbar, counter, total):
        while counter.value < total:
            pbar.n = counter.value
            pbar.refresh()
            time.sleep(0.1)
        pbar.n = total
        pbar.refresh()
    
    pbar = tqdm(total=len(incomplete_df), desc="Processing tasks")

    progress_thread = Thread(target=update_progress_bar, args=(pbar, progress_counter, len(incomplete_df)))
    progress_thread.start()
    
    # Fill task queue
    for _, row in incomplete_df.iterrows():
        task_queue.put(row.to_dict()) 
    
    # Add poison pills
    for _ in range(args.n_procs * args.n_threads):
        task_queue.put(None)
    
    # Create worker arguments
    worker_args = (args.output, args.model, args.reasoning_effort, args.temperature, args.language,
                   args.multimodal, args.max_retries, args.timeout)
    
    # Start multiprocessing
    if args.n_procs == 1:
        process_worker(task_queue, worker_args, args.n_threads, 0, progress_counter, progress_lock)
    else:
        with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
            futures = [executor.submit(process_worker, task_queue, worker_args, args.n_threads, process_id, progress_counter, progress_lock) 
                    for process_id in range(args.n_procs)]
            for future in futures:
                future.result()

    progress_thread.join()
    pbar.close()
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()