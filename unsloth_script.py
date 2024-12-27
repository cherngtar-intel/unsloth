import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default="none", choices=['none', 'all', '1', '2'], help="to set benchmark option")
parser.add_argument("--sanity", action="store_true", help="to set sanity option")
parser.add_argument("--once", action="store_true", help="to run iteration once only")

args = parser.parse_args()

if args.benchmark == 'all': RUN_BENCHMARK_1, RUN_BENCHMARK_2 = True, True
elif args.benchmark == '1': RUN_BENCHMARK_1, RUN_BENCHMARK_2 = True, False
elif args.benchmark == '2': RUN_BENCHMARK_1, RUN_BENCHMARK_2 = False, True
else: RUN_BENCHMARK_1, RUN_BENCHMARK_2 = False, False

RUN_SANITY_TEST = args.sanity

if args.once: num_iter_benchmark, num_iter_sanity = 1, 1
else: num_iter_benchmark, num_iter_sanity = 5, 2

# Define the script to run and the parameters
script_name = "unsloth_inference.py"

param_list_benchmark_1 = [
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "meta-llama/Llama-3.2-3B-Instruct", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "microsoft/Phi-3-mini-4k-instruct", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "google/gemma-2b-it", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "HuggingFaceH4/zephyr-7b-beta", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "warm_up": None, "output_csv": None, "outfile": "benchmark_1.csv"},
    # Add more parameter combinations as needed
]

param_list_benchmark_2 = [
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "meta-llama/Llama-3.2-3B-Instruct", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "microsoft/Phi-3-mini-4k-instruct", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "google/gemma-2b-it", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "HuggingFaceH4/zephyr-7b-beta", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "text_streamer": "custom", "output_csv": None, "outfile": "benchmark_2.csv"},
    # Add more parameter combinations as needed
]

param_list_sanity = [
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "output_csv": None, "outfile": "sanity.csv", "remark": "none"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "warm_up": None, "output_csv": None, "outfile": "sanity.csv", "remark": "warm up"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": None, "output_csv": None, "outfile": "sanity.csv", "remark": "chat template"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "text_streamer": "default", "output_csv": None, "outfile": "sanity.csv", "remark": "streamer:default"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "text_streamer": "custom", "output_csv": None, "outfile": "sanity.csv", "remark": "streamer:custom"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "warm_up": None, "text_streamer": "custom", "output_csv": None, "outfile": "sanity.csv", "remark": "streamer:custom, warm up"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "output_csv": None, "outfile": "sanity.csv", "prompt": "What is AI?", "remark": "diff prompt"},
    # Add more parameter combinations as needed
]

def run(param_list, num_iteration, name):
    # Loop to execute the script with different parameters
    for i, params in enumerate(param_list):
        for j in range(num_iteration):
            print(f"\nRunning {name} : item #{i + 1} : iter #{j + 1} : {params}\n")

            # Constructing the command, use -u for unbuffered output
            command = ["python", "-u", script_name]
            
            # Add parameters to the command
            for key, value in params.items():
                command.append(f"--{key}")
                if value is not None:
                    command.append(value)
                    
            # Run the script and display output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                for line in process.stdout:  # Stream the output line by line
                    print(line, end="")  # Print line to the terminal
            except Exception as e:
                print(f"Error during {name} : item #{i + 1} : iter #{j + 1}: {e}")
            
            # Wait for the process to complete
            process.wait()
            
            # Check if there were any errors
            if process.returncode != 0:
                print(f"{name} : #{i + 1} : iter #{j + 1} : failed with errors:")
                for line in process.stderr:
                    print(line, end="")


# Start trigger running Benchmark/Sanity
if RUN_BENCHMARK_1: run(param_list_benchmark_1, num_iter_benchmark, "Benchmark_1")
if RUN_BENCHMARK_2: run(param_list_benchmark_2, num_iter_benchmark, "Benchmark_2")
if RUN_SANITY_TEST: run(param_list_sanity, num_iter_sanity, "Sanity")

print(f"\nThe end")