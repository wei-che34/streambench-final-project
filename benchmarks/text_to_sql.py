import os
import hashlib
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from colorama import Fore, Style
from .base import Bench
from .text2sql_utils.sqlite_interpreter import execute_model
from .text2sql_utils.string_formatter import generate_schema_prompt

def create_bird():
    class StreamingBird(GeneralText2SQL):
        DATASET_PATH = 'appier-ai-research/StreamBench_public'
        DATASET_NAME = 'bird'
        def __init__(self, db_path='data/bird/dev_databases', **kwargs):
            super().__init__(db_path=db_path, **kwargs)
    return StreamingBird

class GeneralText2SQL(Bench):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    def __init__(
        self,
        split: str = "test",
        db_path: str = None,
        **kwargs
    ) -> None:
        super().__init__(config={})
        self.split = split
        self.db_path = db_path
        self.total = 0
        self.eval_set = self.dataset[self.split]
        self.initialize()
        self.sql_results = []

    def get_dataset(self) -> Dataset:
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.eval_set

    def initialize(self) -> None:
        # print("Initializing DB schema prompts...")
        self.db_prompt_schema = dict()
        for row in tqdm(self.eval_set, dynamic_ncols=True):
            if row["db_id"] not in self.db_prompt_schema:
                db_path = os.path.join(self.db_path, row["db_id"], row["db_id"] + ".sqlite")
                schema_prompt = generate_schema_prompt(db_path)
                self.db_prompt_schema[row["db_id"]] = schema_prompt

    def postprocess_generation(self, generation: str, idx: int = -1) -> str:
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for GeneralText2SQL-Task)
        """
        # TODO: check if the SQL is valid, if not then notify the user
        if not self.check_sql_validity(generation):
            print(Fore.RED + f"Invalid SQL: {generation}" + Style.RESET_ALL)
        return generation

    def check_sql_validity(self, sql_code: str) -> bool:
        """
        Check if the given SQL code is valid.
        """
        # TODO
        return True

    def get_input(self, row: dict) -> dict:
        schema = self.db_prompt_schema[row["db_id"]]
        question = row["question"]
        return {"table_schema": schema, "user_query": question}

    def get_output(self, row: dict):
        return {
            "SQL": row["SQL"], 
            "db_id": row["db_id"], 
            "label": row["SQL"],
            "question_id": row["question_id"]     
        }

    def process_results(self, generations: str, label: dict, return_details: bool = False, **kwargs):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param labels: original labels
            list of str containing refrences
        """
        db_path = os.path.join(self.db_path, label['db_id'], label['db_id'] + '.sqlite')
        res = execute_model(generations, label['SQL'], db_path)
        
        # for kaggle submission
        pred_res = res.get("predicted_res", "")
        self.sql_results.append({
            "id": label["question_id"],
            "result": self.hash_prediction(pred_res)
        })
        
        correct = res.get("res", 0)
        self.n_correct += correct
        self.predictions.append(generations)
        self.references.append(label)
        self.total += 1
        rolling_acc = self.n_correct / self.total
        if return_details:
            return {
                "result": 'Answer is Correct' if correct == 1 else 'Answer is NOT Correct',
                "correct": correct,
                "n_correct": self.n_correct,
                "rolling_acc": rolling_acc
            }
        return correct

    def get_metrics(self):
        return {
            "EX": self.n_correct / self.total
        }

    def give_feedback(self, pred_res: dict) -> bool:
        return bool(pred_res["correct"])

    def normalize_value(self, val):
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return val

    def hash_prediction(self, res):
        if type(res) is list:
            # Normalize each tuple and each element within
            res = [tuple(self.normalize_value(item) for item in t) for t in res]
            res = [str(item) for item in res]
            res = sorted(list(set(res)))
            res = "||".join(res)
        
        # Hash the normalized frozenset
        return hashlib.md5(res.encode()).hexdigest()

    def save_output(self, output_path):
        df = pd.DataFrame(self.sql_results)

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the DataFrame to the specified path
        df.to_csv(output_path, index=False)

# Manual testing
if __name__ == "__main__":
    DB_ROOT_PATH = "../bird-benchmark/data/bird-benchmark/dev/dev_databases"
    DB_ID = "financial"
    db_path = os.path.join(DB_ROOT_PATH, DB_ID, DB_ID + ".sqlite")
    schema_prompt = generate_schema_prompt(db_path)
    # Test get_zeroshot_prompt
    question = "What is the total amount of money spent on all transactions?"
    zeroshot_prompt = GeneralText2SQL.get_zeroshot_prompt(schema_prompt, question)
    # Test get_fewshot_template
    fewshot_text = "SELECT SUM(amount) FROM transactions"
    fewshot_prompt = GeneralText2SQL.get_fewshot_template(schema_prompt, question).format(fewshot_text=fewshot_text)
    print(fewshot_prompt)
