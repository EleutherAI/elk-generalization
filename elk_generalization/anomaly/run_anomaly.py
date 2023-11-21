import os

models = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
]
template_names = ["mixture", "grader_first", "grader_last"]

if __name__ == "__main__":
    for base_model in models:
        for template in template_names:
            model_last = base_model.split("/")[-1]
            quirky_model = f"atmallen/{model_last}-{template}"
            quirky_model_last = quirky_model.split("/")[-1]

            method = "mahalanobis"
            subtract_diag = False

            command = (
                f"python anomaly_experiment.py "
                f"--model {quirky_model_last} --method {method} "
                f"--experiments-dir "
                f"../../experiments "
            )
            if subtract_diag:
                command += "--subtract-diag "
            os.system(command)
