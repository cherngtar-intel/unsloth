from concurrent import futures
import grpc
import unsloth_pb2
import unsloth_pb2_grpc

from unsloth import FastLanguageModel
import torch
import timeit
import time
from transformers import TextStreamer, AutoTokenizer
from unsloth.chat_templates import get_chat_template
import queue
import threading
import gc

# from huggingface_hub import login
# login(token = "hf_...")

class UnslothModelService(unsloth_pb2_grpc.UnslothModelServiceServicer):
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def Init(self, request, context):
        print("Init:", request.model)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = request.model,
            max_seq_length = 2048,
            load_in_4bit = False,
            device_map = "xpu",
            # local_files_only = True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(request.model, trust_remote_code=True)
        FastLanguageModel.for_inference(self.model)
        self.model = self.model.to("xpu")
        return unsloth_pb2.InitResponse(init_status=True)

    def Generate(self, request, context):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # print("request.prompt:", request.prompt)
        inputs = self.tokenizer.encode(request.prompt, return_tensors="pt").to('xpu')

        # Define a queue to store generated tokens
        token_queue = queue.Queue()

        # Define the custom callback function
        def custom_callback(token_id):
            if token_id.ndimension() == 1 and token_id.size(0) == 1:
                token = self.tokenizer.decode(token_id[0], skip_special_tokens=True)
                token_queue.put(token)

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

        output_holder = {}

        def model_generate_wrapper(output_holder, **kwargs):
            result = self.model.generate(**kwargs)
            output_holder['output'] = result

        # Create a generator function
        def generate_text_stream(tokenizer, inputs, max_new_tokens, terminators=None):
            custom_streamer = CustomTextStreamer(tokenizer, custom_callback)
            kwargs = {
                'output_holder': output_holder,
                    'input_ids': inputs,
                    'streamer': custom_streamer,
                    'max_new_tokens': max_new_tokens,
                    'use_cache': True
            }
            if terminators != None:
                kwargs['eos_token_id'] = terminators
            
            thread = threading.Thread(target=model_generate_wrapper, kwargs=kwargs)
            thread.start()
            while thread.is_alive() or not token_queue.empty():
                try:
                    token = token_queue.get(timeout=1)
                    yield token
                except queue.Empty:
                    continue

        kwargs = {
            'tokenizer': self.tokenizer,
            'inputs': inputs,
            'max_new_tokens': 128
        }
        
        if terminators != None:
            kwargs['terminators'] = terminators

        # Use the generator function to generate text and yield tokens
        start_time = time.time()

        timestamp_first_token = 0
        for token in generate_text_stream(**kwargs):
            if timestamp_first_token == 0:
                timestamp_first_token = time.time()
            yield unsloth_pb2.GenerateResponse(generated_text=token)

        end_time = time.time()
        generation_time = end_time - start_time

        num_input_tokens = inputs.shape[1]
        first_token_delay = timestamp_first_token - start_time
        num_output_tokens = 0
        num_generated_tokens = 0
        throughput = 0
        yield unsloth_pb2.GenerateResponse(input_tokens_count=num_input_tokens)

        if 'output' in output_holder:
            num_output_tokens = output_holder['output'].shape[1]
            num_generated_tokens = num_output_tokens - num_input_tokens
            throughput = num_generated_tokens / generation_time
            throughput_2p = (num_generated_tokens-1)/(end_time-timestamp_first_token)

        yield unsloth_pb2.GenerateResponse(output_tokens_count=num_generated_tokens)

        print(f"### first_token_delay: {first_token_delay:.4f}")
        print(f"### num_input_tokens: {num_input_tokens}")
        print(f"### num_output_tokens: {num_output_tokens}")
        print(f"### num_generated_tokens: {num_generated_tokens}")
        print(f"### Inference_time: {generation_time:.4f} seconds")
        print(f"### Throughput: {throughput:.2f} tokens/second")
        print(f"### Throughput_2+: {throughput_2p:.2f} tokens/second")

    def UnloadLLM(self, request, context):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.xpu.empty_cache()
        print("Model unloaded, memory freed.")

        return unsloth_pb2.UnloadLLMResponse(status=True)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    unsloth_pb2_grpc.add_UnslothModelServiceServicer_to_server(UnslothModelService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("Server started on port 50052")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()