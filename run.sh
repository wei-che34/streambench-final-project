bench_names=(
    "classification_public"
    "classification_private"
    "sql_generation_public"
    "sql_generation_private"
)
model_name="prince-canuma/Ministral-8B-Instruct-2410-HF"
device="cuda:1"

echo -e "\033[0;36mRunning Zero-Shot:\033[0m"
for bench_name in ${bench_names[@]}; do
    echo -e "\033[0;32mRunning benchmark: $bench_name\033[0m"
    python -m examples.zeroshot --bench_name $bench_name --model_name $model_name --device $device
done

echo -e "\033[0;36mRunning Self-StreamICL:\033[0m"
for bench_name in ${bench_names[@]}; do
    echo -e "\033[0;32mRunning benchmark: $bench_name\033[0m"
    python -m examples.self_streamicl --bench_name $bench_name --model_name $model_name --device $device
done
