import io
import os

from openai import OpenAI
import numpy as np

from calibrate_thresh import analyze_dataset

if __name__ == "__main__":
    MODEL = "gpt-4"
    REQUIRED_FRAC_CORRECT = 0.96
    BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims, and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. The input is: "

    ALPHA = 0.2
    CONFIDENCE_METHOD = "optimal"

    ALPHAS = np.arange(0.05, 0.80, 0.05)
    CONFIDENCE_METHODS_RAW = [
        "random",
        "baseline",
        "gpt",
        "frequency",
        #    "frequency+gpt",
        "optimal",
    ]
    CONFIDENCE_METHODS_RANKING = [
        "random-ranking",
        "baseline-ranking",
        "gpt-ranking",
        "frequency-ranking",
        #    "frequency+gpt-ranking",
        "optimal-ranking",
    ]

    # Get Open AI key.
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key (with command: export OAI_KEY=[OAI_KEY])"
        )
    OPENAI_CLIENT = OpenAI(api_key=OAI_KEY)

    # Load names from Factscore dataset.
    with io.open("data/factscore_names.txt", "r") as fopen:
        factscore_names = fopen.readlines()[:]

    dataset_prefix = "factscore"
    analyze_dataset(
        dataset_prefix,
        factscore_names,
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
        compute_single_threshold=True,
        create_plots=True,
        create_histogram=False,
        calib=False,
    )
