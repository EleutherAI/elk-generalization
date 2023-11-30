import os

models = [
    "atmallen/pythia-410m",
    "atmallen/pythia-1b",
    "atmallen/pythia-1.4b",
    "atmallen/pythia-2.8b",
    "atmallen/pythia-6.9b",
    "atmallen/pythia-12b",
    "atmallen/Llama-2-7b-hf",
    "atmallen/Mistral-7B-v0.1",
]
template_names = ["mixture", "grader_first", "grader_last"]

if __name__ == "__main__":
    for reporter in ["lr", "mean-diff", "lda", "lr-on-pair", "ccs", "crc"]:
    # for reporter in ["mean-diff", "lda", "lr-on-pair", "ccs", "crc"]:
        for base_model in models:
            for template in template_names:
                model_last = base_model.split("/")[-1]
                quirky_model_last = f"{model_last}-{template}"

                method = "mahalanobis"
                subtract_diag = True

                command = (
                    f"python anomaly_experiment.py "
                    f"--model {quirky_model_last} --method {method} "
                    f"--reporter {reporter} "
                    f"--experiments-dir "
                    f"../../experiments "
                )
                if subtract_diag:
                    command += "--subtract-diag "
                os.system(command)
