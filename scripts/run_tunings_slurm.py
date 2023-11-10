import subprocess
import os
from pathlib import Path

# Datasets
datasets = ["abalone",
     "adult", "buddy", "california", "cardio",
    "churn2", "default", "diabetes", "fb-comments", "gesture",
    "higgs-small", "house", "insurance", "king", "miniboone", "wilt",
]

cmd_base = ("KBGEN_LOGDIR=~/ python scripts/tune_ddpm.py {data} synthetic catboost kbformer "
            "--device {device} --eval_seeds")

# SLURM job
slurm_job = """#!/bin/zsh
#SBATCH --job-name={data}-kbformer
#SBATCH --output=logs/{data}-%j.out
#SBATCH --error=logs/{data}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,iaifi_gpu

conda activate tddpm
"""

# RUN THE JOB for each dataset
slurm_scripts_path = Path("logs/scripts")
os.makedirs(slurm_scripts_path, exist_ok=True)

for dataset in datasets:
    job = slurm_job.format(data=dataset) + cmd_base.format(data=dataset, device="cuda:0")
    job_name = slurm_scripts_path / f"tune_{dataset}.sh"
    with open(job_name, "w") as f:
        f.write(job)
    subprocess.run(["sbatch", job_name])