"""
Example running from the root of the project:
```
python eval/eval_mednli.py --model_address http://127.0.0.1:8080 --model_name_or_path meta-llama/Meta-Llama-3-8B --model_has_system --model_is_instruct
```
"""
import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

from utils import build_few_shot_examples, build_model_input, update_results, compute_average_results

sys_prompt = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will recieve two sentences, labeled 'SENTENCE_1' and 'SENTENCE_2', respectively. Your task is to determine the logical relation between the two sentences. Valid answers are: ENTAILMENT, NEUTRAL or CONTRADICTION.""" 

user_prompt_template = """
SENTENCE_1: {sentence1}
SENTENCE_2: {sentence2}
"""

assistant_response_template =  """{gold_label}"""

ground_truth_key = "gold_label"


def compute_metrics(model_output, label):
    """
    Retrieves the label of the given model output, uppercases it, and then finds
    all occurrences of the label in the input using regular expressions. If only
    one occurrence is found, the function returns an accuracy of 1, otherwise it
    returns an accuracy of 0.

    Args:
        model_output (str): 1-hot encoded output of the machine learning model,
            which is searched for labels to determine the accuracy of the model's
            predictions.
        label (str): 10-class label of the image, which is used to filter and
            determine the accuracy of the generated documentation.

    Returns:
        dict: a dictionary containing either "ACCURACY" with a value of 1 or 0,
        indicating the model's accuracy.

    """
    answer = re.findall(label.upper(), model_output.upper())
    if len(answer) == 1:
        return {"ACCURACY": 1}
    return {"ACCURACY": 0}
    

def main():

    """
    Parses command-line arguments and prepares data for few-shot learning tasks
    using a pre-trained model. It loads data, builds few-shot examples, and uses
    an OpenAI API to generate responses to the input prompt. The function computes
    metrics and writes them to log files, and also saves the results in a JSON file.

    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_address", type=str)
    argument_parser.add_argument("--model_name_or_path", type=str)
    argument_parser.add_argument("--model_has_system", action='store_true')
    argument_parser.add_argument("--model_is_instruct", action='store_true')
    argument_parser.add_argument("--num_few_shot_examples", type=int, default=1)
    # argument_parser.add_argument("--data_path", type=str, default="data/MedNLI/mli_train_v1.jsonl")
    argument_parser.add_argument("--data_path", type=str, default="data/MedNLI/mli_dev_v1.jsonl")
    argument_parser.add_argument("--log_path", type=str, default="logs")
    argument_parser.add_argument("--token", type=str)
    args = argument_parser.parse_args()

    # # Tokenizer & Inference client
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    # inference_client = InferenceClient(model=args.model_address)
    inference_client = None

    # Load data
    with open(args.data_path, "r") as data_file:
        data = [json.loads(line) for line in data_file]

    log_path = Path(args.log_path)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
        
    if (log_path / "results.json").exists():
        print(f"Skipping dataset {args.data_path} as results already exist")
        return


    if (log_path / "predictions.json").exists():
        (log_path / "predictions.json").unlink()

    # Create few shot examples
    few_shot_chat = build_few_shot_examples(
        data[:args.num_few_shot_examples],
        sys_prompt,user_prompt_template, assistant_response_template, args.model_has_system, args.model_is_instruct
    )

    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
        api_key = "sk-no-key-required"
    )

    results = {}
    for i, entry in enumerate((pbar := tqdm(data[args.num_few_shot_examples:]))):
        model_input = build_model_input(entry, user_prompt_template, args.model_is_instruct, few_shot_chat, tokenizer=tokenizer)
        model_input += assistant_response_template.format(**{ground_truth_key: ""})
        
        if i == 0:
            # Print first model input to log format
            with open(log_path / "debug_model_input.txt", "w") as f_w:
                f_w.write(model_input)

        ground_truth = entry[ground_truth_key]

        if "llama-3" in args.model_name_or_path.lower() or "llama3" in args.model_name_or_path.lower():
            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": model_input}
                ]
            )
            output = completion.choices[0].message.content
            if "<|im_end|>" in output:
                output = output.split("<|im_end|>")[0]
        else:
            output = inference_client.text_generation(
                model_input,
                max_new_tokens=20,
                stream=False,
                details=False
            )
        
        # Cut off new self-prompting
        output = re.sub(
            r"(You are an AI.*)|(\[INST\].*)|((<\|user\|>).*)", "", output)
        

        # Update metric variables
        new_results = compute_metrics(output, ground_truth)
        update_results(results, new_results)
        average_results = compute_average_results(results)
        
        # Print metrics
        print_metrics = ["ACCURACY"]
        pbar.set_description(
            ", ".join(f"Average {k}: {v:.2f}" for k, v in average_results.items() if k in print_metrics))

        new_results["Model Answer"] = output
        new_results["Ground Truth Answer"] = ground_truth
        with open(log_path / "predictions.json", "a") as out_file:
            json.dump(new_results, out_file)
            out_file.write("\n")

    with open(log_path / "results.json", "w") as f_w:
        json.dump(average_results, f_w)


if __name__ == "__main__":
    main()
