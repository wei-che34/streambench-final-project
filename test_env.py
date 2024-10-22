import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tqdm import tqdm
from benchmarks import TASKS
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

task_key = {
    'classification_public': 'accuracy',
    'sql_generation_public': 'EX',
}

def print_header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 50}")
    print(f"{text.center(50)}")
    print(f"{'=' * 50}{Style.RESET_ALL}")

def print_task_result(task, metrics, key):
    result = "PASSED" if metrics[key] > 0.99 else "FAILED"
    color = Fore.GREEN if result == "PASSED" else Fore.RED
    print(f"{Fore.YELLOW}Task: {Style.BRIGHT}{task}")
    print(f"{Fore.BLUE}Metric: {Style.BRIGHT}{key}")
    print(f"{Fore.BLUE}Score: {Style.BRIGHT}{metrics[key]:.4f}")
    print(f"{color}Result: {Style.BRIGHT}{result}")
    print(f"{Style.RESET_ALL}{'-' * 30}")

if __name__ == "__main__":
    print_header("Environment Setup Validation")
    
    for task, cls_ in TASKS.items():
        print(f"\n{Fore.MAGENTA}Processing task: {Style.BRIGHT}{task}{Style.RESET_ALL}")
        bench = cls_()
        for time_step, row in enumerate(tqdm(bench.get_dataset(), dynamic_ncols=True, 
                                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            label = bench.get_output(row)
            if task in ['sql_generation_public']:
                pred = label['SQL']
            else:
                pred = label
            pred_res = bench.process_results(
                pred,
                label,
                return_details=True,
                time_step=time_step
            )
        metrics = bench.get_metrics()
        print_task_result(task, metrics, task_key[task])
        
    print_header("Validation Complete")
