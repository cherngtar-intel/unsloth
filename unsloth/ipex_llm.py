#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#import os
#os.environ['HTTP_PROXY'] = "http://proxy-dmz.intel.com:912"
#os.environ['HTTPS_PROXY'] = "http://proxy-dmz.intel.com:912"

import torch
import time
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from ipex_llm import optimize_model

#from huggingface_hub import login
#login(token = "hf_....")

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(user_input: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    prompt_texts = [f'<|begin_of_text|>']

    if system_prompt != '':
        prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    return ''.join(prompt_texts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama3 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", # meta-llama/Meta-Llama-3-8B-Instruct | meta-llama/Llama-3.2-3B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct
                        help='The huggingface repo id for the Llama3 (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128, #32
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 use_cache=True,)

    # With only one line to enable IPEX-LLM optimization on model
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
#    model = optimize_model(model)
    model = optimize_model(model, low_bit='fp16')

    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}, Data type: {param.dtype}")

    model = model.to('xpu')
    #model = model.half().to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # here the terminators refer to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct#transformers-automodelforcausallm
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    
    #args.prompt = "Describe the tallest tower in the world."

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        num_input_tokens = input_ids.size(1)

        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                eos_token_id=terminators,
                                max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        output = model.generate(input_ids,
                                eos_token_id=terminators,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        num_output_tokens = output.size(1)
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        #print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output (skip_special_tokens=False)', '-'*20)
        print(output_str)
        num_generated_tokens = num_output_tokens - num_input_tokens
        generation_time = end - st
        throughput = num_generated_tokens / generation_time
        print(f"num_input_tokens : {num_input_tokens}")
        print(f"num_output_tokens : {num_output_tokens}")
        print(f"Generated Tokens: {num_generated_tokens}")
        print(f"Inference time: {generation_time:.4f} seconds")
        print(f"Throughput: {throughput:.2f} tokens/second")
