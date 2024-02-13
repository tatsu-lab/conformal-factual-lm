import io
import os

from openai import OpenAI
import numpy as np

from calibrate_thresh import analyze_dataset

if __name__ == "__main__":
    MODEL = "gpt-4"
    REQUIRED_FRAC_CORRECT = 1
    BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims (make sure not to add any information), and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. If the input is short, it is fine to only return 1 claim. The input is: "

    # Global variables for computing single treshold to appear in f'out/nq_a={REQUIRED_FRAC_CORRECT}_{CONFIDENCE_METHOD}_threshold.txt'
    ALPHA = 0.1
    CONFIDENCE_METHOD = "frequency+gpt"

    # Global variables for producing conformal plots to appear in f'out/nq_raw_a={REQUIRED_FRAC_CORRECT}_fig.png and out/nq_ranking_a={REQUIRED_FRAC_CORRECT}_fig.png'
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
        #    "frequency-ranking",
        "frequency+gpt-ranking",
        "optimal-ranking",
    ]

    # Get Open AI key.
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key (with command: export OAI_KEY=[OAI_KEY])"
        )
    OPENAI_CLIENT = OpenAI(api_key=OAI_KEY)

    def nq_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
        )
        return f"You will get a natural question and parts of an answer, which you are to merge into coherent prose. Make sure to include all the parts in the answer. There may be parts that are seemingly unrelated to the others, but DO NOT add additional information or reasoning to merge them. \n\nThe parts:\n{claim_string}\n\nThe question:\n{prompt}. Remember, DO NOT add any additional information or commentary, just combine the parts."

    # Load questions from nq dataset.
    with io.open("data/nq_questions.txt", "r") as fopen:
        nq_questions = fopen.readlines()[20:]

    dataset_prefix = "nq"
    analyze_dataset(
        dataset_prefix,
        nq_questions,
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
        merge_prompt=nq_merge_prompt,
        create_histogram=False,
        calib=False,
    )
