import os
import numpy as np
from run_elk import elk_reporter_dir
from pathlib import Path

model = "atmallen/pythia-410m-v11665991"
template_name = "grader_first"

alphas = 10**np.linspace(-4.0, 4.0, num=9, endpoint=True)
l1_ratios = np.linspace(0.0, 1.0, num=5, endpoint=True)

def main():
    original_elk_dir = Path(elk_reporter_dir())
    try:
        for alpha in alphas:
            alpha = np.round(alpha, 10)
            for l1_ratio in l1_ratios:
                os.environ["ELK_DIR"] = str(original_elk_dir / f"alpha={alpha}_l1_ratio={l1_ratio}")
                command = f"python run_elk.py --model {model} --template {template_name} " \
                          f"--max-examples 4096 1024 --num-gpus 2 --alpha {alpha} --l1-ratio {l1_ratio}"
                print(f"RUNNING: {command}")
                os.system(command)

                # command = f"python run_elk.py --model {model} --template {template_name} --max-examples 4096 1024 --num-gpus -1 --experiment easy_vs_hard"
                # print(f"RUNNING: {command}")
                # os.system(command)
    finally:
        os.environ["ELK_DIR"] = str(original_elk_dir)

if __name__ == "__main__":
    main()

