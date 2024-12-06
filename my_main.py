from base import Agent
from execution_pipeline import main

from utils import RAG, strip_all_lines
import re
import random
from colorama import Fore, Style
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
from transformers import logging as transformers_logging
import gc

class LocalModelAgent(Agent):
    """
    A base agent that uses a local model for text generation tasks.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize the local model
        """
        super().__init__(config)
        self.llm_config = config
        self.model_paths = {
            1: config["model_1"],
            2: config["model_2"],
            3: config["model_3"]
        }
        self.tokenizers = {
            1: AutoTokenizer.from_pretrained(config["model_1"]),
            2: AutoTokenizer.from_pretrained(config["model_2"]),
            3: AutoTokenizer.from_pretrained(config["model_3"])
        }
        self.models = {}
        self.rag = RAG(config["rag"])
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()

        # Preload all models to CPU
        for agent_num, model_path in self.model_paths.items():
            if self.llm_config["use_8bit"]:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_has_fp16_weight=False
                )
                self.models[agent_num] = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=quantization_config,
                    device_map="cpu"
                )
            else:
                self.models[agent_num] = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu"
                )
            self.models[agent_num].eval()

    def move_model_to_device(self, agent_num: int, device: str):
        """
        Move the specified model to the given device (e.g., "cuda" or "cpu").
        """
        self.models[agent_num].to(device)

    def generate_response(self, messages: list, agent_num: int) -> str:
        """
        Generate a response using the local model of the corresponding agent_num.
        """
        # Move the required model to GPU
        self.move_model_to_device(agent_num, device=self.llm_config["device"])
        model = self.models[agent_num]
        tokenizer = self.tokenizers[agent_num]

        text_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text_chat], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Move the model back to CPU after usage
        self.move_model_to_device(agent_num, device="cpu")
        torch.cuda.empty_cache()

        return response    

    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """
        if correctness:
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
            return True
        return False

class ClassificationAgent(LocalModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = """\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile.
        Provide your diagnosis in the following format: <number>. <diagnosis>""".strip()
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(
        option_text: str,
        text: str
    ) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the following patient profile:

        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Now, directly provide the diagnosis for the patient in the following format: <number>. <diagnosis>""".strip()
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        {{question}}
        Diagnosis: {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        option_text: str,
        text: str,
    ) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Here are some example cases.
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        {text}        
        
        Now provide the diagnosis for the patient in the following format: <number>. <diagnosis>"""
        return strip_all_lines(prompt)

    def decide_agent_num(self) -> int:
        return random.randint(1, 3)

    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        self.reset_log_info()
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)
        
        shots = self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot

        combined_prompt = f"{system_prompt}\n{prompt}"
        messages = [
            {"role": "user", "content": combined_prompt}
        ]

        agent_num = self.decide_agent_num()
        print(f"Using agent {agent_num} for classification.")

        response = self.generate_response(messages, agent_num)
        prediction = self.extract_label(response, label2desc)
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizers[agent_num].encode(system_prompt + prompt)),
            "num_output_tokens": len(self.tokenizers[agent_num].encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": response,
        })
        self.inputs.append(text)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        return prediction

    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                if int(numbers[0]) in label2desc:
                    prediction = numbers[0]
                else:
                    print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                    prediction = random.choice(list(label2desc.keys()))
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL    code at the current time step.
        """
        # TODO
        raise NotImplementedError
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--model_1', type=str, default="google/gemma-2-9b-it")
    parser.add_argument('--model_2', type=str, default="mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument('--model_3', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='path to save csv file for kaggle submission')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    if args.bench_name.startswith("classification"):
        max_tokens = 16
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = 512
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    config = {
        'bench_name': bench_cfg['bench_name'],
        'model_1': args.model_1,
        'model_2': args.model_2,
        'model_3': args.model_3,
        'device': args.device,
        'exp_name': f'MAM_streamicl_{args.bench_name}_{args.model_1}_{args.model_2}_{args.model_3}',
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            'embedding_model': 'BAAI/bge-base-en-v1.5',
            'seed': 42,
            "top_k": 4,
            "order": "similar_at_top"
        }
    }
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    agent = agent_name(config)
    main(agent, bench_cfg)
