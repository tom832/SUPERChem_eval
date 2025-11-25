# Evaluation Scripts

This directory contains scripts for evaluating Large Language Model (LLM) outputs. The evaluation is divided into three main tasks:
1.  **Main Evaluation**: Running a model on the benchmark to generate answers.
2.  **Checkpoint Ability Tagging**: Running various model checkpoints on a dataset to generate answers and tag their capabilities.
3.  **Chain-of-Thought (CoT) Evaluation**: Using a powerful "judge" LLM to perform a fine-grained evaluation of a model's reasoning steps.

---

## 1. Main Evaluation

This is the primary process for generating model answers for the benchmark questions.

### Files
- [`eval.sh`](eval.sh): The main shell script to configure and run the evaluation.
- `eval.py`: The Python script that handles the logic of interacting with the models.
- `prompt_en.txt`: The English prompt template used to query the models.

### How It Works
The [`eval.sh`](eval.sh) script reads a model configuration and executes `eval.py` to generate answers for a given question set. It is designed for a single, comprehensive evaluation run.

### Usage
1.  **Configure [`eval.sh`](eval.sh)**:
    -   Set the `INPUT_FILE` variable to the path of your questions file.
    -   Set the `MODEL` variable to the checkpoint you want to test.
    -   Configure other parameters like temperature as needed.
2.  **Run the script**:
    ```bash
    bash eval.sh
    ```
3.  **Check the results**: The output file will be saved in the `../data/` directory.

---

## 2. Checkpoint Ability Tagging

This process is designed to assess and label the abilities of different model checkpoints. It runs a set of questions through each specified model and generates corresponding answers.

### Files
- [`eval_ckpt.sh`](eval_ckpt.sh): The main shell script to configure and run the evaluation.
- `eval_ckpt.py`: The Python script that handles the logic of interacting with the models.
- `prompt_en_ckpt.txt`: The English prompt template used to query the models.

### How It Works
The [`eval_ckpt.sh`](eval_ckpt.sh) script reads a list of model configurations. For each model, it executes `eval_ckpt.py` to generate answers for a given question set (`.parquet` file). The output is tagged based on the model's performance and the abilities being tested.

### Usage
1.  **Configure [`eval_ckpt.sh`](eval_ckpt.sh)**:
    -   Set the `INPUT_FILE` variable to the path of your questions file.
    -   Modify the `MODELS` array to include the checkpoints you want to test, along with their parameters (temperature, pass@k, etc.).
2.  **Run the script**:
    ```bash
    bash eval_ckpt.sh
    ```
3.  **Check the results**: The output files will be saved in the `../data/` directory. Each filename will contain details about the model, temperature, and other parameters used for the run.

---

## 3. Chain-of-Thought (CoT) Evaluation

This process uses a strong LLM as a "judge" to evaluate the quality of another model's generated answers, with a specific focus on the correctness of its Chain-of-Thought reasoning.

### Files
- [`eval_cot.sh`](eval_cot.sh): The main shell script to configure and run the CoT evaluation.
- `eval_cot.py`: The Python script that sends requests to the judge model.
- `prompt_en_cot.txt`: The English prompt template that instructs the judge model on how to perform the evaluation.

### How It Works
The [`eval_cot.sh`](eval_cot.sh) script automatically discovers model answer files (`.jsonl`) in the project's root directory. For each file, it invokes `eval_cot.py`, which calls the specified judge model (e.g., Gemini 2.5 Pro). The judge model analyzes the reasoning steps in the answer file and provides a score or evaluation.

### Usage
1.  **Place answer files**: Ensure the `.jsonl` answer files you want to evaluate are present in the project's root directory.
2.  **Configure [`eval_cot.sh`](eval_cot.sh)**:
    -   Set the `EVALUATOR_MODEL` to the powerful LLM you want to use as the judge.
    -   Set `QUESTIONS_FILE` to the original question data file.
    -   Set `ANSWERS_PREFIX` to match the prefix of your answer files.
3.  **Run the script**:
    ```bash
    bash eval_cot.sh
    ```
4.  **Check the results**: The evaluation results will be saved in the `../data/` directory. The script appends `_EVAL_BY_<JudgeModelName>` to the original answer filename.