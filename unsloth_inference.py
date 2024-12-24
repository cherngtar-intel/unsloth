from unsloth import FastLanguageModel
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="meta-llama/Llama-3.2-1B-Instruct", help="to set model path")
parser.add_argument("--chat_template", action="store_true", help="to use Unsloth's chat template")
parser.add_argument("--text_streamer", action="store_true", help="to use text streamer")
parser.add_argument("--warm_up", action="store_true", help="to perform the first warm up attempt")
parser.add_argument("--max_token", default=128, type=int, help="max generated token")
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
max_tokens = args.max_token
prompt = args.prompt
print("args:", args)
print("model:", model_name)

if prompt == "input":
    prompt = input('Prompt: ')

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

# warm up
if USE_WARMUP:
    print("warm up...")
    _ = model.generate(**kwargs)

# text streamer
if USE_TEXT_STREAMER:
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    kwargs['streamer'] = text_streamer

# model generation
start_time = time.time()
output = model.generate(**kwargs)
torch.xpu.synchronize()
end_time = time.time()
output = output.cpu()
if not USE_TEXT_STREAMER:
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    print(output_str)

# benchmark metric
num_input_tokens = inputs.size(1)
num_output_tokens = output.size(1)
num_generated_tokens = num_output_tokens - num_input_tokens
generation_time = end_time - start_time
throughput = num_generated_tokens / generation_time
print(f"### Total Input Tokens : {num_input_tokens}")
print(f"### Total Output Tokens : {num_output_tokens}")
print(f"### Generated Tokens: {num_generated_tokens}")
print(f"### Inference Time: {generation_time:.4f} seconds")
print(f"### Throughput: {throughput:.4f} tokens/second")
