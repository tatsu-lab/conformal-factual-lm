import json
import numpy as np

if __name__ == "__main__":
    merged_filename = "out/factscore_merged_a=1_alpha=0.2_conf=frequency+gpt.jsonl"
    original_filename = "data/factscore_annotations.jsonl"

    with open(merged_filename, "r") as fopen:
        merged = json.load(fopen)["data"]

    with open(original_filename, "r") as fopen:
        original = json.load(fopen)["data"]

    merged = merged[-30:]
    original = original[-30:]
    choice = np.random.randint(1, 3, size=len(merged))
    data = []

    for i in range(len(merged)):
        c_factual_output = merged[i]["new-output"]
        gpt4_output = original[i]["original-output"]
        c_factual_output_option = choice[i]

        if c_factual_output_option == 1:
            data.append(
                {
                    "Answer 1": c_factual_output,
                    "Answer 2": gpt4_output,
                    "C-factal option": str(c_factual_output_option),
                }
            )
        else:
            data.append(
                {
                    "Answer 1": gpt4_output,
                    "Answer 2": c_factual_output,
                    "C-factal option": str(c_factual_output_option),
                }
            )

    with open("out/factscore_survey.json", "w") as outfile:
        merged_json = {"data": data}
        json.dump(merged_json, outfile, indent=4)
