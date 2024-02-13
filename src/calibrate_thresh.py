import json
from math import ceil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import rankdata
from sayless import (
    get_frequency_scores,
    get_subclaims,
    merge_subclaims,
    query_model,
    default_merge_prompt,
)
import random

CORRECT_ANNOTATIONS = ["Y", "S"]


def get_legend_name(confidence_method):
    if (
        confidence_method == "frequency+gpt"
        or confidence_method == "frequency+gpt-ranking"
    ):
        return "Frequency"
    elif confidence_method == "baseline" or confidence_method == "baseline-ranking":
        return "Ordinal"
    elif confidence_method == "gpt" or confidence_method == "gpt-ranking":
        return "GPT-4 confidence"
    elif confidence_method == "random" or confidence_method == "random-ranking":
        return "Random"
    elif confidence_method == "optimal" or confidence_method == "optimal-ranking":
        return "Optimal"


def get_title_name(dataset_prefix):
    if dataset_prefix == "factscore":
        return "FActScore"
    elif dataset_prefix == "nq":
        return "NQ"
    elif dataset_prefix == "MATH":
        return "MATH"
    return dataset_prefix


def dump_claims(output_list, filename="claims.jsonl"):
    """
    Dumps output_list into filename.
    [{"prompt": "Who is Tatsu?", "claims": [{"subclaim": "Tatsu is Japanese person", 'correct': 1.0}, {"subclaim": "Tatsu was born in 1988", 'correct': 0.0} ..]}]
    """
    with open(filename, "w") as outfile:
        merged_json = {"data": output_list}
        json.dump(merged_json, outfile, indent=4)


def load_calibration(filename="claims.jsonl"):
    """
    Reverse of dump_claims.
    """
    with open(filename, "r") as fopen:
        return json.load(fopen)["data"]


def get_ranking(entry, confidence_method, use_percent=True):
    """
    Returns the corresponding ranking scores from the raw scores of confidence_method.
    """
    score_list = [
        -(subclaim[confidence_method + "-score"] + subclaim["noise"])
        for subclaim in entry["claims"]
    ]
    rankings = len(entry["claims"]) + 1 - rankdata(score_list, method="ordinal")
    if use_percent:
        rankings = rankings / len(entry["claims"])
    return rankings


def get_confidence(entry, method, openai_client, model):
    """
    Takes in an entry from {}_annotations.jsonl and returns a list of confidence scores from method.
    """
    if method == "random":
        return [np.random.normal(0, 1) for subclaim in entry["claims"]]
    elif method == "baseline":
        return [
            len(entry["claims"]) - x for x in list(range(1, len(entry["claims"]) + 1))
        ]
    elif method == "gpt":
        return [float(subclaim["gpt-score"]) for subclaim in entry["claims"]]
    elif method == "frequency":
        return get_frequency_scores(
            openai_client, entry["claims"], entry["prompt"], 5, model
        )
    # This assumes frequency was already added.
    elif method == "frequency+gpt":
        return [
            subclaim["gpt-score"] + subclaim["frequency-score"]
            for subclaim in entry["claims"]
        ]
    elif method == "optimal":
        return [
            int(subclaim["annotation"] in CORRECT_ANNOTATIONS)
            for subclaim in entry["claims"]
        ]
    # This assumes the corresponding raw scores were already added.
    elif method in [
        "random-ranking",
        "baseline-ranking",
        "gpt-ranking",
        "frequency-ranking",
        "frequency+gpt-ranking",
        "optimal-ranking",
    ]:
        return get_ranking(
            entry, method[:-8]
        )  # -8 is to remove '-ranking' from method.
    else:
        print(f"{method} method is not implemented.")


def add_scores(calibration_data, filename, confidence_methods, openai_client, model):
    """
    Adds noise (to break ties later) and scores for each method in confidence_methods to filename.
    """
    # Add a random draw to the data if it does not already exist
    if "noise" not in calibration_data[0]["claims"][0]:
        for entry in tqdm(calibration_data):
            for i, output in enumerate(entry["claims"]):
                output["noise"] = np.random.normal(0, 0.001)

        # Write to file if any modification was made.
        dump_claims(calibration_data, filename)

    # If confidence_method is not already computed, compute and add it to calibration_data.
    for confidence_method in confidence_methods:
        if confidence_method + "-score" not in calibration_data[0]["claims"][0]:
            print(f"Computing {confidence_method} method")
            for entry in tqdm(calibration_data):
                score_list = get_confidence(
                    entry, confidence_method, openai_client, model
                )
                for i, output in enumerate(entry["claims"]):
                    output[confidence_method + "-score"] = score_list[i]

        # Write to file if any modification was made.
        dump_claims(calibration_data, filename)

    return calibration_data


def get_r_score(entry, confidence_method, a):
    """
    Compute the r_a score for entry when confidence_method is used as the sub-claim scoring function.
    """
    threshold_set = sorted(
        [
            subclaim[confidence_method + "-score"] + subclaim["noise"]
            for subclaim in entry["claims"]
        ],
        reverse=True,
    )
    curr_threshold = threshold_set[0]
    for threshold in threshold_set:
        curr_threshold = threshold
        # Apply threshold.
        accepted_subclaims = [
            subclaim
            for subclaim in entry["claims"]
            if subclaim[confidence_method + "-score"] + subclaim["noise"] >= threshold
        ]

        # Compute entailed/correct fraction.
        entailed_fraction = (
            np.mean(
                [
                    subclaim["annotation"] in CORRECT_ANNOTATIONS
                    for subclaim in accepted_subclaims
                ]
            )
            if accepted_subclaims
            else 1
        )

        if entailed_fraction < a:
            return curr_threshold
    return -100000  # -100000 is less than any score assigned by any of the implemented confidence methods


def compute_threshold(alpha, calibration_data, a, confidence_method):
    """
    Computes the quantile/threshold from conformal prediction.
    # alpha: float in (0, 1)
    # calibration_data: calibration data
    # a: as in paper, required fraction correct
    # confidence_method: string
    """
    # Compute r score for each example.
    r_scores = [get_r_score(entry, confidence_method, a) for entry in calibration_data]

    # Compute threshold for conformal prection. The quantile is ceil((n+1)*(1-alpha))/n, and
    # We map this to the index by dropping the division by n and subtracting one (for zero-index).
    quantile_target_index = ceil((len(r_scores) + 1) * (1 - alpha))
    threshold = sorted(r_scores)[quantile_target_index - 1]
    return threshold


def create_correctness_vs_removed_plot(
    dataset_prefix, data, alphas, a, confidence_methods, fig_filename
):
    """
    Creates leave-one-out conformal plots.
    """
    print(f"Producing conformal plot: {fig_filename}")
    plt.figure(dpi=800)

    for confidence_method in tqdm(confidence_methods):
        results = []  # first indexes into alpha, then list of (correct, frac_removed)_i

        for alpha in alphas:
            results_for_alpha = [[], []]
            for i in range(len(data)):
                calibration_data = data[:i] + data[i + 1 :]
                test_data = data[i]

                threshold = compute_threshold(
                    alpha, calibration_data, a, confidence_method
                )
                accepted_subclaims = [
                    subclaim
                    for subclaim in test_data["claims"]
                    if subclaim[confidence_method + "-score"] + subclaim["noise"]
                    >= threshold
                ]
                fraction_removed = 1 - len(accepted_subclaims) / len(
                    test_data["claims"]
                )
                entailed_fraction = (
                    np.mean(
                        [
                            subclaim["annotation"] in CORRECT_ANNOTATIONS
                            for subclaim in accepted_subclaims
                        ]
                    )
                    if accepted_subclaims
                    else 1
                )
                correctness = entailed_fraction >= a
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)

            results.append(results_for_alpha)

        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

        # Add standard error.
        yerr = [
            np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
            for results_for_alpha in results
        ]
        label = get_legend_name(confidence_method)

        plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)

    # Plot base factuality point.
    x_point = x[-1]
    y_point = y[-1]
    point_size = 235
    plt.scatter(
        x_point,
        y_point,
        color="black",
        marker="*",
        s=point_size,
        label="Base factuality",
        zorder=1000,
    )

    font_size = 16
    legend_font_size = 13
    dataset_title_name = get_title_name(dataset_prefix)
    if a == 1:
        plt.title(dataset_title_name, fontsize=font_size + 4)
        plt.xlabel("Fraction of factual outputs", fontsize=font_size)
    else:
        plt.title(f"{dataset_title_name}, a={a}", fontsize=font_size + 4)
        plt.xlabel(f"Fraction achieving avg factuality >= {a}")
    plt.ylabel("Average percent removed", fontsize=font_size)

    legend = plt.legend(
        loc="upper left", bbox_to_anchor=(1, 1), fontsize=legend_font_size
    )
    legend.get_title().set_fontsize(legend_font_size)
    plt.savefig(fig_filename, bbox_inches="tight")


def create_calibration_plot(
    dataset_prefix, data, alphas, a, confidence_method, fig_filename
):
    """
    Creates calibration plot.
    """
    print(f"Producing calibration plot: {fig_filename}")
    fig, ax = plt.subplots(figsize=(6, 4))

    results = []  # first indexes into alpha. then list of (correct, frac_removed)_i

    for alpha in tqdm(alphas):
        results_for_alpha = [[], []]
        for i in range(1000):
            # Randomly shuffle the data
            random.shuffle(data)

            # Split the data into two equal parts
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]

            threshold = compute_threshold(alpha, calibration_data, a, confidence_method)

            accepted_subclaim_list = [
                [
                    subclaim
                    for subclaim in test_data_point["claims"]
                    if subclaim[confidence_method + "-score"] + subclaim["noise"]
                    >= threshold
                ]
                for test_data_point in test_data
            ]
            entailed_fraction_list = [
                (
                    np.mean(
                        [
                            subclaim["annotation"] in CORRECT_ANNOTATIONS
                            for subclaim in accepted_subclaims
                        ]
                    )
                    if accepted_subclaims
                    else 1
                )
                for accepted_subclaims in accepted_subclaim_list
            ]
            correctness_list = [
                entailed_fraction >= a for entailed_fraction in entailed_fraction_list
            ]
            fraction_correct = sum(correctness_list) / len(correctness_list)
            results_for_alpha[0].append(1 - alpha)
            results_for_alpha[1].append(fraction_correct)

        results.append(results_for_alpha)

    x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
    y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
    # yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

    print(x)
    print(y)
    # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")

    x_values = np.linspace(0.3, 0.98, 100)

    # Plot lower bound.
    y_values = x_values
    plt.plot(
        x_values, y_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
    )

    # Plot upper bound
    y_values = x_values + 1 / (split_index + 1)
    plt.plot(x_values, y_values, "--", color="gray", linewidth=2)
    plt.plot(x, y, label=get_title_name(dataset_prefix), linewidth=2)

    plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
    plt.legend()
    plt.ylabel("Empirical factuality", fontsize=16)
    plt.savefig(fig_filename, bbox_inches="tight", dpi=800)


def generate_merged_outputs(
    data,
    alpha,
    a,
    confidence_method,
    openai_client,
    model,
    merged_filename,
    merge_prompt,
):
    """
    Creates jsonl file with original output and new suclaims.
    """
    print(
        f"Merging accepted subclaims for a={a}, alpha={alpha} and confidence_method={confidence_method}"
    )
    for i in tqdm(range(len(data))):
        calibration_data = data[:i] + data[i + 1 :]
        test_data = data[i]

        threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
        accepted_subclaims = [
            subclaim
            for subclaim in test_data["claims"]
            if subclaim[confidence_method + "-score"] + subclaim["noise"] >= threshold
        ]
        test_data["new-output"] = merge_subclaims(
            openai_client,
            accepted_subclaims,
            model,
            test_data["prompt"],
            create_merge_prompt=merge_prompt,
        )
        test_data["all-subclaims"] = [
            {"subclaim": subclaim["subclaim"], "annotation": subclaim["annotation"]}
            for subclaim in test_data["claims"]
        ]
        test_data["accepted-subclaims"] = [
            {"subclaim": subclaim["subclaim"], "annotation": subclaim["annotation"]}
            for subclaim in accepted_subclaims
        ]

    merged_data = (
        [
            {
                "prompt": entry["prompt"],
                "original-output": entry["original-output"],
                "all-subclaims": entry["all-subclaims"],
                "accepted-subclaims": entry["accepted-subclaims"],
                "new-output": entry["new-output"],
            }
            for entry in data
        ]
        if "original-output" in calibration_data[0]
        else [
            {
                "prompt": entry["prompt"],
                "all-subclaims": entry["all-subclaims"],
                "accepted-subclaims": entry["accepted-subclaims"],
                "new-output": entry["new-output"],
            }
            for entry in data
        ]
    )

    dump_claims(merged_data, merged_filename)


def create_hist(
    dataset_prefix,
    data,
    alpha,
    a,
    confidence_method,
    openai_client,
    model,
    hist_filename,
    merge_prompt,
):
    """
    Creates histogram showing the fraction of subclaims removed across all outputs.
    """
    print(
        f"Creating histogram for a={a}, alpha={alpha} and confidence_method={confidence_method}"
    )
    plt.figure(dpi=800)
    fraction_removed_list = []
    for i in tqdm(range(len(data))):
        calibration_data = data[:i] + data[i + 1 :]
        test_data = data[i]
        threshold = compute_threshold(alpha, calibration_data, a, confidence_method)

        accepted_subclaims = [
            subclaim
            for subclaim in test_data["claims"]
            if subclaim[confidence_method + "-score"] + subclaim["noise"] >= threshold
        ]
        fraction_removed = 1 - len(accepted_subclaims) / len(test_data["claims"])
        fraction_removed_list.append(fraction_removed)

    fontsize = 15
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plt.xlabel("Percent removed", fontsize=fontsize)
    plt.ylabel("Fraction of outputs", fontsize=fontsize)
    plt.title(
        f"{get_title_name(dataset_prefix)}, {chr(945)}={alpha}", fontsize=fontsize
    )
    weights = np.ones_like(fraction_removed_list) / float(len(fraction_removed_list))
    plt.hist(fraction_removed_list, weights=weights)
    plt.savefig(hist_filename, bbox_inches="tight", dpi=800)


def analyze_dataset(
    dataset_prefix,
    input_dataset,
    openai_client,
    model,
    breakdown_prompt,
    confidence_method,
    confidence_methods_raw,
    confidence_methods_ranking,
    alpha,
    alphas,
    a,
    compute_single_threshold=True,
    merge=False,
    create_plots=True,
    create_histogram=False,
    calib=False,
    merge_prompt=default_merge_prompt,
):
    """
    Performs the desired analysis for a given dataset.
    """
    # Generate outputs and subclaims if they do not exist, in {dataset_prefix}_subclaims.jsonl file that
    # can be (manually) copied over to /data/{dataset_prefix}_annotations.jsonl and then annotated.
    if not os.path.exists(f"data/{dataset_prefix}_annotations.jsonl"):
        print(
            f"Creating dataset for annotation. When done, please copy out/{dataset_prefix}_subclaims.jsonl to data/{dataset_prefix}_annotations.jsonl and annotate."
        )
        data = []
        # Generate outputs for each prompt
        for prompt in tqdm(input_dataset):
            output = query_model(openai_client, prompt, model)

            # Extract subclaims. "annotation" field is automtically set to 'N'.
            subclaims = get_subclaims(
                openai_client, output, model, breakdown_prompt=breakdown_prompt
            )
            claim_list = [
                {
                    "subclaim": subclaim["subclaim"],
                    "gpt-score": subclaim["gpt-score"],
                    "annotation": "N",
                }
                for subclaim in subclaims
            ]
            data.append(
                {"prompt": prompt, "original-output": output, "claims": claim_list}
            )
        dump_claims(data, f"out/{dataset_prefix}_subclaims.jsonl")

    else:
        # Otherwise, get the annotated subclaims add all scores if they are not already there.
        if not os.path.exists(f"out/{dataset_prefix}_subclaims_with_scores.jsonl"):
            print(
                f"Computing scores for subclaims. These will appear in out/{dataset_prefix}_subclaims_with_scores.jsonl"
            )
            calibration_data = load_calibration(
                f"data/{dataset_prefix}_annotations.jsonl"
            )
            add_scores(
                calibration_data,
                f"out/{dataset_prefix}_subclaims_with_scores.jsonl",
                confidence_methods_raw + confidence_methods_ranking,
                openai_client,
                model,
            )

        calibration_data = load_calibration(
            f"out/{dataset_prefix}_subclaims_with_scores.jsonl"
        )

        if compute_single_threshold:
            # Compute a single threshold for ALPHA and CONFIDENCE_METHOD.
            threshold = compute_threshold(
                alpha,
                calibration_data,
                a,
                confidence_method,
            )
            with open(
                f"out/{dataset_prefix}_a={a}_alpha={alpha}_conf={confidence_method}.txt",
                "w",
            ) as fopen:
                fopen.write(str(threshold))

        if create_histogram:
            create_hist(
                dataset_prefix,
                calibration_data,
                alpha,
                a,
                confidence_method,
                openai_client,
                model,
                f"out/{dataset_prefix}_hist_a={a}_alpha={alpha}_conf={confidence_method}.png",
                merge_prompt,
            )

        if merge:
            generate_merged_outputs(
                calibration_data,
                alpha,
                a,
                confidence_method,
                openai_client,
                model,
                f"out/{dataset_prefix}_merged_a={a}_alpha={alpha}_conf={confidence_method}.jsonl",
                merge_prompt,
            )

        if create_plots:
            # Create plots for ALPHAS, CONFIDENCE_METHODS_RAW, CONFIDENCE_METHODS_RANKING.
            create_correctness_vs_removed_plot(
                dataset_prefix,
                calibration_data,
                alphas,
                a,
                confidence_methods_raw,
                f"out/{dataset_prefix}_raw_a={a}_fig.png",
            )
            create_correctness_vs_removed_plot(
                dataset_prefix,
                calibration_data,
                alphas,
                a,
                confidence_methods_ranking,
                f"out/{dataset_prefix}_ranking_a={a}_fig.png",
            )

        if calib:
            # Create calibration plots for ALPHAS, CONFIDENCE_METHODS_RAW, CONFIDENCE_METHODS_RANKING.
            create_calibration_plot(
                dataset_prefix,
                calibration_data,
                alphas,
                a,
                confidence_method,
                f"out/{dataset_prefix}_raw_calibration_a={a}.png",
            )
