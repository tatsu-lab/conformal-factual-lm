# Language Models with Conformal Factuality Guarantees

This is the official repository for the paper Language Models with Conformal Factuality Guarantees by [Christopher Mohri](https://scholar.google.com/citations?user=_otSGXcAAAAJ) and [Tatsunori Hashimoto](https://thashim.github.io/). 

## Installation
Install the packages in `requirements.txt` and also run the pre-commit script `pre-commit install`.

## Files
- `main.py`: shows an example of running inference on one example 
- `sayless.py`: core functions necessary for inference (sub-claim splitting + merging)
- `factscore.py`: code for FActScore dataset
- `nq.py`: code for Natural Questions dataset
- `math_data.py`: code for MATH dataset
- `calibrate_thresh.py`: code for analyzing each dataset (producing sub-claims for annotation, computing thresholds, producing plots, generating new outputs)

## Simple FActScore example
The script `factscore.py` will first dump a jsonl with sub-claims into `out/factscore_subclaims.jsonl`.
Afterwards, the user will need to copy the file to `data/factscore_annotations.jsonl` and annotate the correctness entries in the json.
Then, the user can compute conformal thresholds, produce any of the plots appearing in the paper, and generate conformally factual outputs by setting the appropriate fields in `analyze_dataset`.
`nq.py` and `math_data.py` have the same functionality but for the Natural Questions and MATH datasets.
