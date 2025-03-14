from unsloth import FastLanguageModel
import torch
import time
import argparse
import unsloth_shared as shared

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="meta-llama/Llama-3.2-1B-Instruct", help="to set model path")
parser.add_argument("--chat_template", action="store_true", help="to use Unsloth's chat template")
parser.add_argument("--text_streamer", default="none", choices=['none', 'default', 'custom'], help="to use text streamer")
parser.add_argument("--warm_up", action="store_true", help="to perform the first warm up attempt")
parser.add_argument("--max_token", default=128, type=int, help="to set max generated token")
parser.add_argument("--output_csv", action="store_true", help="to dump output to csv file")
parser.add_argument("--outfile", type=str, default="benchmark.csv", help="to save csv output with input file name")
parser.add_argument("--remark", type=str, default="", help="to add some remark into csv output data")
parser.add_argument('--prompt', type=str, default="Describe the tallest tower in the world.", help="to input prompt for inference")
args = parser.parse_args()

model_name = args.model

### Llama ###
## model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
## model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
## model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
## model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

### Mistral ###
## model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
## model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"

### Phi-3 ###
## model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
## model_name = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
# model_name = "microsoft/Phi-3-mini-4k-instruct"

### Gemma ###
## model_name = "unsloth/gemma-2b-bnb-4bit"
## model_name = "unsloth/gemma-2-2b-bnb-4bit"
## model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
## model_name = "google/gemma-7b-it"
## model_name = "google/gemma-2-2b-it"
# model_name = "google/gemma-2b-it"

### Zephyr ###
## model_name = "unsloth/zephyr-sft-bnb-4bit"
## model_name = "HuggingFaceH4/mistral-7b-sft-beta"
# model_name = "HuggingFaceH4/zephyr-7b-beta"

### Qwen ###
## model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
# model_name = "Qwen/Qwen2.5-3B-Instruct"


USE_CHAT_TEMPLATE = args.chat_template
USE_TEXT_STREAMER = args.text_streamer
USE_WARMUP = args.warm_up
OUTPUT_CSV = args.output_csv
output_file= args.outfile
output_remark = args.remark
max_tokens = args.max_token
prompt = args.prompt

print("args:", args)
print("model:", model_name)

if prompt == "input":
    prompt = input('Prompt: ')

load_start_time = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = False, # original value = True
    device_map = "xpu",
    # local_files_only = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

if USE_CHAT_TEMPLATE:

    from unsloth.chat_templates import get_chat_template

    model_path = model_name.lower()
    if "llama" in model_path: template = "llama-3.1"
    elif "mistral" in model_path: template = "mistral"
    elif "phi-3" in model_path: template = "phi-3"
    elif "gemma" in model_path: template = "gemma"
    elif "zephyr" in model_path: template = "zephyr"
    elif "qwen2.5" in model_path: template = "qwen2.5"
    print("chat_template:", template)
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = template,
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )

    messages = [ {"role": "human", "content": prompt}, ]
    # messages = [ {"role": "human", "content": "Describe the tallest tower in the world."}, ]
    # messages = [ {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"}, ]
    # messages = [ {"from": "human", "value": "What is Unsloth?"}, ]
    # messages = [ {"from": "human", "value": "What is AI?"}, ]

    terminators = None
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("xpu")

else:

    # Note: use USE_CHAT_TEMPLATE = False to workaround for Phi-3, Mistral, Gemma problematic chat template
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def get_prompt(user_input: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
        prompt_texts = [f'<|begin_of_text|>']

        if system_prompt != '':
            prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

        for history_input, history_response in chat_history:
            prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
            prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
        return ''.join(prompt_texts)

    messages = prompt
    # messages = "Describe the tallest tower in the world."
    # messages = "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"
    # messages = "What is Unsloth?"
    # messages = "What is AI?"    

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    DEFAULT_SYSTEM_PROMPT = """\
    """

    prompt = get_prompt(messages, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

pass

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
model = model.to("xpu")
#model = model.half().to('xpu')

kwargs = { 'input_ids': inputs, 'max_new_tokens': max_tokens, 'use_cache': True }
if terminators != None:
    kwargs['eos_token_id'] = terminators

load_end_time = time.time()

# warm up
if USE_WARMUP:
    print("warming up...")
    _ = model.generate(**kwargs)

# text streamer
if USE_TEXT_STREAMER == "default":

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    kwargs['streamer'] = text_streamer

elif USE_TEXT_STREAMER == "custom":

    from transformers import TextStreamer
    first_valid_token_time = 0

    # Define the custom callback function
    def custom_callback(token_id):
        if token_id.ndimension() == 1 and token_id.size(0) == 1:
            global first_valid_token_time
            if first_valid_token_time == 0:
                first_valid_token_time = time.time()
            # token = tokenizer.decode(token_id[0], skip_special_tokens=True)

    # Subclass TextStreamer to create a custom streamer
    class CustomTextStreamer(TextStreamer):
        def __init__(self, tokenizer, callback):
            super().__init__(tokenizer)
            self.callback = callback

        def put(self, token_id):
            if isinstance(token_id, list):
                for tid in token_id:
                    super().put(tid)  # Call the original put method for each token ID
                    self.callback(tid)  # Call the custom callback for each token ID
            else:
                super().put(token_id)  # Call the original put method
                self.callback(token_id)  # Call the custom callback

    custom_streamer = CustomTextStreamer(tokenizer, custom_callback)
    kwargs['streamer'] = custom_streamer

# model generation
generation_start_time = time.time()
output = model.generate(**kwargs)
torch.xpu.synchronize()
generation_end_time = time.time()
output = output.cpu()
if USE_TEXT_STREAMER == "none":
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)

# benchmark metric
num_input_tokens = inputs.size(1)
num_output_tokens = output.size(1)
num_generated_tokens = num_output_tokens - num_input_tokens
first_token_latency_ms = shared.first_token_delay_time * 1000
loading_time = load_end_time - load_start_time
generation_time = generation_end_time - generation_start_time
throughput = num_generated_tokens / generation_time
max_memory = torch.xpu.max_memory_allocated()

print(f"### Input Tokens Count: {num_input_tokens}")
print(f"### Output Tokens Count: {num_output_tokens}")
print(f"### Generated Tokens Count: {num_generated_tokens}")
print(f"### Loading Time: {loading_time:.6f} secs")
print(f"### Inference Time: {generation_time:.6f} secs")
print(f"### 1st Token Latency: {first_token_latency_ms:.6f} msec")
print(f"### Throughput: {throughput:.6f} tokens/sec")
print(f"### Max memory allocated: {max_memory / (1024 ** 3):02} GB")

if USE_TEXT_STREAMER == "custom":
    throughput_2p = (num_generated_tokens - 1) / (generation_end_time - first_valid_token_time)
    first_valid_token_latency_ms = (first_valid_token_time - generation_start_time) * 1000
    print(f"### 1st Valid Token Latency: {first_valid_token_latency_ms:.6f} msec")
    print(f"### Throughput 2+: {throughput_2p:.6f} tokens/sec")


if OUTPUT_CSV:
    # Set up data for CSV metrics output
    current_time = time.time()
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
    output_data = {
        'Timestamp': formatted_time,
        'model_name:' : model_name,
        'Input Tokens Count': num_input_tokens,
        'Output Tokens Count': num_output_tokens,
        'Generated Tokens Count': num_generated_tokens,
        'Loading Time (s)': round(loading_time, 6),
        'Inference Time (s)': round(generation_time, 6),
        '1st Token Latency (ms)': round(first_token_latency_ms, 6),
        'Throughput (token/s)': round(throughput, 6),
        '1st Valid Token Latency (ms)': None,
        'Throughput 2+ (token/s)': None,
        'Remark': output_remark
    }

    if USE_TEXT_STREAMER == "custom":
        output_data['1st Valid Token Latency (ms)'] = round(first_valid_token_latency_ms,6)
        output_data['Throughput 2+ (token/s)'] = round(throughput_2p,6)

    # Write data to a CSV file, append to existing file or create new file given path
    import os
    import csv
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.abspath(os.path.join(current_script_dir, output_file))
    file_exists = os.path.isfile(csv_file_path)

    if not file_exists:
        try: open(csv_file_path, 'w')
        except: print("Failed to create file: ", csv_file_path)

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=output_data.keys())
        if not file_exists:
            writer.writeheader() # Add header if none
        writer.writerow(output_data) # Append data as new row
        print(f"Metrics dumped to {csv_file_path}.")
