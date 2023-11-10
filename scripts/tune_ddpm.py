import subprocess
import lib
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("ds_name", type=str)
parser.add_argument("eval_type", type=str)
parser.add_argument("eval_model", type=str)
parser.add_argument("prefix", type=str)
parser.add_argument("--eval_seeds", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()
ds_name = args.ds_name
eval_type = args.eval_type
assert eval_type in ("merged", "synthetic")
prefix = str(args.prefix)

pipeline = f"scripts/pipeline.py"
base_config_path = f"exp/{ds_name}/config.toml"
parent_path = Path(f"exp/{ds_name}/")
exps_path = Path(
    f"exp/{ds_name}/many-exps/"
)  # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f"scripts/eval_seeds.py"

os.makedirs(exps_path, exist_ok=True)


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 0.005)
    num_decoder_mixtures = trial.suggest_int("num_decoder_mixtures", 1, 100, log=True)
    d_model = trial.suggest_int("d_model", 4, 9)
    d_model = 2**d_model
    depth = trial.suggest_int("depth", 1, 2)

    batch_size = trial.suggest_categorical("batch_size", [8192])
    # init_var = trial.suggest_loguniform("init_var", 1e-2, 2)
    init_var = 1
    steps = 5_000
    # steps = trial.suggest_categorical("steps", [10])  # for debug

    # sampling params
    # tempreture = trial.suggest_uniform("tempreture", 0.8, 1.2)
    # leaps = trial.suggest_int("leaps", 1, 5)

    base_config = lib.load_config(base_config_path)

    base_config["train"]["main"]["lr"] = lr
    base_config["train"]["main"]["steps"] = steps
    base_config["train"]["main"]["batch_size"] = batch_size
    base_config["train"]["main"]["weight_decay"] = 0.0
    base_config["model_params"]["rtdl_params"][
        "num_decoder_mixtures"
    ] = num_decoder_mixtures
    base_config["model_params"]["rtdl_params"]["d_model"] = d_model
    base_config["model_params"]["rtdl_params"]["init_var"] = init_var
    base_config["model_params"]["rtdl_params"]["num_layers"] = depth
    base_config["model_params"]["rtdl_params"]["field_encoder_layers"] = depth
    base_config["model_params"]["rtdl_params"]["field_decoder_layers"] = depth
    base_config["eval"]["type"]["eval_type"] = eval_type
    
    base_config["sample"]["num_samples"] = min(base_config["sample"]["num_samples"], 50_000)
    base_config["sample"]["batch_size"] = 50_000
    # base_config["sample"]["leaps"] = leaps
    # base_config["sample"]["temperature"] = tempreture

    base_config["device"] = args.device
    base_config["parent_dir"] = str(exps_path / f"{trial.number}")
    base_config["eval"]["type"]["eval_model"] = args.eval_model
    if args.eval_model == "mlp":
        base_config["eval"]["T"]["normalization"] = "quantile"
        base_config["eval"]["T"]["cat_encoding"] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / "config.toml")

    subprocess.run(
        [
            "python3.9",
            f"{pipeline}",
            "--config",
            f'{exps_path / "config.toml"}',
            "--train",
            "--change_val",
        ],
        check=True,
    )

    n_datasets = 5
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config["sample"]["seed"] = sample_seed
        lib.dump_config(base_config, exps_path / "config.toml")

        subprocess.run(
            [
                "python3.9",
                f"{pipeline}",
                "--config",
                f'{exps_path / "config.toml"}',
                "--sample",
                "--eval",
                "--change_val",
            ],
            check=True,
        )

        report_path = str(
            Path(base_config["parent_dir"]) / f"results_{args.eval_model}.json"
        )
        report = lib.load_json(report_path)

        # TODO check if we want to optimize r2 or f1
        if "r2" in report["metrics"]["val"]:
            score += report["metrics"]["val"]["r2"]
        else:
            score += report["metrics"]["val"]["macro avg"]["f1-score"]

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

best_config_path = parent_path / f"{prefix}_best/config.toml"
best_config = study.best_trial.user_attrs["config"]
best_config["parent_dir"] = str(parent_path / f"{prefix}_best/")

os.makedirs(parent_path / f"{prefix}_best", exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(
    optuna.importance.get_param_importances(study),
    parent_path / f"{prefix}_best/importance.json",
)

subprocess.run(
    [
        "python3.9",
        f"{pipeline}",
        "--config",
        f"{best_config_path}",
        "--train",
        "--sample",
    ],
    check=True,
)

if args.eval_seeds:
    best_exp = str(parent_path / f"{prefix}_best/config.toml")
    subprocess.run(
        [
            "python3.9",
            f"{eval_seeds}",
            "--config",
            f"{best_exp}",
            "10",
            "ddpm",
            eval_type,
            args.eval_model,
            "5",
        ],
        check=True,
    )
