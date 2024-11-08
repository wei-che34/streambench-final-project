import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
from tqdm import tqdm
from colorama import Fore, Style

from utils import merge_dicts
from benchmarks import load_benchmark, Bench

def main(agent, bench_cfg, debug: bool = False, debug_samples: int = 10, use_wandb: bool = False, wandb_name: str = None, wandb_config: dict = None):
    bench_cfg['agent'] = agent
    # bench_cfg['agent_callback'] = agent.retrieve_experience
    print('init bench environment')
    bench: Bench = load_benchmark(bench_cfg['bench_name'])(**bench_cfg)
    agent.bench = bench

    ds = bench.get_dataset()
    if debug:
        print(Fore.YELLOW + f"Debug mode: using first {debug_samples} samples" + Style.RESET_ALL)
        ds = ds.select(range(debug_samples))

    if use_wandb:
        import wandb
        wandb.init(
            project=f"ADL-StreamBench-{bench_cfg['bench_name']}",
            name=wandb_name,
            config=wandb_config
        )

    pbar = tqdm(ds, dynamic_ncols=True)
    for time_step, row in enumerate(pbar):
        row['time_step'] = time_step
        x = bench.get_input(row)
        model_output = agent(**x)
        prediction = bench.postprocess_generation(model_output, time_step)
        label = bench.get_output(row)
        pred_res = bench.process_results(
            prediction,
            label,
            return_details=True,
            time_step=time_step
        )

        correctness = bench.give_feedback(pred_res)
        agent.update(correctness)

        if use_wandb:
            wandb.log(data=merge_dicts([agent.get_wandb_log_info(), pred_res]))

        if isinstance(label, int):
            label = bench.LABEL2TEXT[label]
        elif isinstance(label, dict):
            label = label.get("label", json.dumps(label))
        agent.log(label_text=label)

        # Update rolling accuracy in tqdm
        pbar.set_description(f"Step {time_step} | Rolling Accuracy: {pred_res['rolling_acc'] * 100:.2f}%")
        pbar.update(1)
    pbar.close()

    metrics = bench.get_metrics()
    print(metrics)
    if use_wandb:
        wandb.log(data={f"final/{k}": v for k, v in metrics.items()})

    output_path = bench_cfg.get("output_path", None)
    if output_path is not None:
        bench.save_output(output_path)

    return metrics

if __name__ == "__main__":
    main()
