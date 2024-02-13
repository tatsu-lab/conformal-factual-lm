import io
import json
import os

from openai import OpenAI
import numpy as np

from calibrate_thresh import analyze_dataset
from datasets import load_dataset


ALPHAS = np.arange(0.05, 0.3, 0.05)
CONFIDENCE_METHODS_RAW = [
    "random",
    "baseline",
    "gpt",
    # "frequency",
    "frequency+gpt",
    "optimal",
]
CONFIDENCE_METHODS_RANKING = [
    "random-ranking",
    "baseline-ranking",
    "gpt-ranking",
    #   "frequency-ranking",
    "frequency+gpt-ranking",
    "optimal-ranking",
]

MODEL = "gpt-4"
REQUIRED_FRAC_CORRECT = 1
BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims (make sure not to add any information), and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. If the input is short, it is fine to only return 1 claim. The input is: "

ALPHA = 0.1
CONFIDENCE_METHOD = "optimal"
DATASET_PREFIX = "MATH"


if __name__ == "__main__":

    def math_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
        )
        return f"You will get a math problem and a set of steps that are true. Construct an answer using ONLY the steps provided. Make sure to include all the steps in the answer, and do not add any additional steps or reasoning. These steps may not fully solve the problem, but merging them could assist someone in solving the problem. \n\nThe steps:\n{claim_string}\n\nThe math problem:\n{prompt}. Remember, do not do any additional reasoning, just combine the given steps."

    # Get Open AI key.
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key (with command: export OAI_KEY=[OAI_KEY])"
        )
    OPENAI_CLIENT = OpenAI(api_key=OAI_KEY)

    # Load questions from math
    dataset = load_dataset("competition_math")
    input_dataset = [question["problem"] for question in dataset["test"]][0:50]
    with io.open("out/MATH.json", "w") as fopen:
        json.dump(dataset["test"][0:50], fopen, indent=4)

    analyze_dataset(
        DATASET_PREFIX,
        input_dataset,
        OPENAI_CLIENT,
        MODEL,
        BREAKDOWN_PROMPT,
        CONFIDENCE_METHOD,
        CONFIDENCE_METHODS_RAW,
        CONFIDENCE_METHODS_RANKING,
        ALPHA,
        ALPHAS,
        REQUIRED_FRAC_CORRECT,
        merge=False,
        create_plots=True,
        compute_single_threshold=False,
        create_histogram=False,
        merge_prompt=math_merge_prompt,
        calib=False,
    )
