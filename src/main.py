import os

from openai import OpenAI

from sayless import query_model, say_less

if __name__ == "__main__":
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key (with command: export OAI_KEY=[OAI_KEY])"
        )
    openai_client = OpenAI(api_key=OAI_KEY)
    prompt = "Tell me a paragraph bio of Percy Liang"
    model = "gpt-4"
    output = query_model(openai_client, prompt, model)
    # Copied threshold from /factscore_a=1_alpha=0.15_conf=frequency+gpt.txt.
    # Compute new ones by running factscore.py with desired parameters and setting compute_single_threshold=True
    threshold = 4.8998812119930735
    merged_output, (accepted_subclaims, all_subclaims) = say_less(
        openai_client, prompt, model, output, threshold
    )
    print("Original output: ")
    print(output)
    print("\n\n\n\n\nModified output: ")
    print(merged_output)
    print("\n\n\n\nAccepted sub-claims: ")
    print(accepted_subclaims)
    print("\n\n\n\nAll sub-claims: ")
    print(all_subclaims)
