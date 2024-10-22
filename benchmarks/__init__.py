from .base import Bench
from .ddxplus import create_ddxplus
from .text_to_sql import create_bird

classes = locals()

TASKS = {
    "classification_public": create_ddxplus(),
    "sql_generation_public": create_bird()
}

def load_benchmark(benchmark_name) -> Bench:
    if benchmark_name in TASKS:
        return TASKS[benchmark_name]
    if benchmark_name in classes:
        return classes[benchmark_name]

    raise ValueError("Benchmark %s not found" % benchmark_name)
