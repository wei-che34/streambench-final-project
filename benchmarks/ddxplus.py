import re
import pandas as pd
import os
import evaluate
from datasets import Dataset
from colorama import Fore, Style

from .base import Bench

class MedicalDiagnosisBench(Bench):
    """A task whose x == patient profile and y == diagnosis."""
    LABEL2TEXT = dict()
    TEXT2LABEL = dict()
    NUM_SHOTS = 16

    def __init__(
        self,
        split: str = "test",
        **kwargs
    ) -> None:
        super().__init__({})
        self.split = split

    def get_dataset(self) -> Dataset:
        return self.dataset[self.split]

    def get_fewshot_prompt(
        self,
        profile: str,
        option_text: str,
    ) -> str:
        fewshot_template = self.get_fewshot_template(profile, option_text)
        return re.sub(r"\{fewshot_text\}", self.fewshot_text, fewshot_template)

    def get_fewshot_text(self) -> str:
        shot_rows = self.dataset["validate"].shuffle(seed=self.seed)
        shots = list()
        for i in range(self.NUM_SHOTS):
            shot = self.get_shot_template().format(
                question=shot_rows[i]["PATIENT_PROFILE"].strip(),
                answer=self.get_label_text(shot_rows[i]["PATHOLOGY"])
            )
            shots.append(shot)
        return "\n\n\n".join(shots).replace("\\", "\\\\")

    def get_input(self, row: dict) -> dict:
        profile = row["PATIENT_PROFILE"].strip()
        return {"label2desc": self.LABEL2TEXT, "text": profile}

    def get_output(self, row: dict) -> dict:
        label_text = row["PATHOLOGY"].lower().strip()
        assert label_text in self.TEXT2LABEL
        return self.TEXT2LABEL[label_text]

    def get_metrics(self) -> dict:
        accuracy = evaluate.load("accuracy")
        metrics = accuracy.compute(predictions=self.predictions, references=self.references)
        return metrics        

    def postprocess_generation(self, res: str, idx: int = -1) -> int:
        number = int(res)
        if number not in self.LABEL2TEXT:
            # Notify the user
            print(Fore.RED + f"Prediction {res} not found in the label set. Please check your output format." + Style.RESET_ALL)
        return number

    def process_results(
        self,
        prediction: str,
        label: str,
        return_details: bool = False,
        **kwargs
    ) -> bool | dict:
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param labels: original labels
            list of str containing refrences
        """
        correct = prediction == label
        self.n_correct += correct
        self.predictions.append(prediction)
        self.references.append(label)

        if return_details:
            return {
                'correct': int(correct),
                'n_correct': self.n_correct,
                'rolling_acc': self.n_correct / len(self.references)
            }
        return correct

    def give_feedback(self, pred_res: dict) -> bool:
        return bool(pred_res["correct"])

    def get_label_text(self, label: int | str) -> str:
        label_int = self.NOTINLABEL
        label_str = "I'm not confident about the diagnosis."
        if isinstance(label, int) and (label in self.LABEL2TEXT):
            label_int = label
            label_str = self.LABEL2TEXT[label_int]
        elif isinstance(label, str) and (label.lower() in self.TEXT2LABEL):
            label_int = self.TEXT2LABEL[label.lower()]
            label_str = label
        return f"{label_int}. {label_str}"

    def save_output(self, output_path):
        df = pd.DataFrame({
            "id": [i for i in range(len(self.predictions))],
            "result": self.predictions
        })

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)

def create_ddxplus():
    class DDXPlusBench(MedicalDiagnosisBench):
        DATASET_PATH = "appier-ai-research/StreamBench_public"
        DATASET_NAME = "ddxplus"
        
        LABEL2TEXT = {
            0: 'Acute COPD exacerbation / infection',
            1: 'Acute dystonic reactions',
            2: 'Acute laryngitis',
            3: 'Acute otitis media',
            4: 'Acute pulmonary edema',
            5: 'Acute rhinosinusitis',
            6: 'Allergic sinusitis',
            7: 'Anaphylaxis',
            8: 'Anemia',
            9: 'Atrial fibrillation',
            10: 'Boerhaave',
            11: 'Bronchiectasis',
            12: 'Bronchiolitis',
            13: 'Bronchitis',
            14: 'Bronchospasm / acute asthma exacerbation',
            15: 'Chagas',
            16: 'Chronic rhinosinusitis',
            17: 'Cluster headache',
            18: 'Croup',
            19: 'Ebola',
            20: 'Epiglottitis',
            21: 'GERD',
            22: 'Guillain-Barré syndrome',
            23: 'HIV (initial infection)',
            24: 'Influenza',
            25: 'Inguinal hernia',
            26: 'Larygospasm',
            27: 'Localized edema',
            28: 'Myasthenia gravis',
            29: 'Myocarditis',
            30: 'PSVT',
            31: 'Pancreatic neoplasm',
            32: 'Panic attack',
            33: 'Pericarditis',
            34: 'Pneumonia',
            35: 'Possible NSTEMI / STEMI',
            36: 'Pulmonary embolism',
            37: 'Pulmonary neoplasm',
            38: 'SLE',
            39: 'Sarcoidosis',
            40: 'Scombroid food poisoning',
            41: 'Spontaneous pneumothorax',
            42: 'Spontaneous rib fracture',
            43: 'Stable angina',
            44: 'Tuberculosis',
            45: 'URTI',
            46: 'Unstable angina',
            47: 'Viral pharyngitis',
            48: 'Whooping cough'
        }
        NOTINLABEL = len(LABEL2TEXT)
        TEXT2LABEL = {v.lower(): k for k, v in LABEL2TEXT.items()}
        LABEL_SET = {v.lower() for v in LABEL2TEXT.values()}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return DDXPlusBench
