import argparse
import os
import pandas as pd
import multiprocessing
import subprocess
import pickle
import json
import time
from tqdm import tqdm

from webppl.templates.model_template import MODEL_TEMPLATE
from webppl.templates.box_ensemble_template import BOX_ENSEMBLE_TEMPLATE

DIR_EXPERIMENTS = "experiments"
DIR_CODEX = "codex"
DIR_MODELS = "models"
DIR_RESULTS = "results"
DIR_WEBPPL = "webppl"

def populate_template(df, task_id, n_simulations: int = 10, n_participants: int = 20, physical_gaussian_pos_noise: float = 3, physical_gaussian_size_noise: float = 3):

    conditional_str = df.loc[task_id, f"codex_conditional_str"]
    box_ensemble_index = df.loc[task_id, f"box_ensemble_index"]

    box_ensemble = BOX_ENSEMBLE_TEMPLATE[box_ensemble_index]

    model_template = MODEL_TEMPLATE.format(
        N_SIMULATIONS=n_simulations, 
        N_PARTICIPANTS=n_participants, 
        PHYSICAL_GAUSSIAN_POS_NOISE=physical_gaussian_pos_noise,
        PHYSICAL_GAUSSIAN_SIZE_NOISE=physical_gaussian_size_noise,
        CONDITIONAL=conditional_str,
        BOX_ENSEMBLE=box_ensemble,
    )

    return model_template

def write_model_for_task(df, task_id, n_simulations, n_participants, physical_gaussian_pos_noise, physical_gaussian_size_noise, model_dir):
    model_template = populate_template(
        df, 
        task_id, 
        n_simulations=n_simulations, 
        n_participants=n_participants, 
        physical_gaussian_pos_noise=physical_gaussian_pos_noise, 
        physical_gaussian_size_noise=physical_gaussian_size_noise
    )
    model_file = os.path.join(model_dir, f"model_{task_id:03d}.wppl")
    with open(model_file, "w") as f:
        f.write(model_template)

def run_single_task(
    task_id,
    model_dir,
    results_dir,
    use_cached: bool = False,
    timeout: int = 120,
    seed: int = 123,
):
    model_file = os.path.join(model_dir, f"model_{task_id:03d}.wppl")
    results_file = os.path.join(results_dir, f"results_{task_id:03d}.pkl")

    # Optionally, load cached prior result
    if use_cached and os.path.exists(results_file):
        with open(results_file, "rb") as f:
            completed_process = pickle.load(f)
        if completed_process is not None:
            print(f"Loaded cached {results_file}")
            return completed_process

    # Run with shell=True
    cmd = f"webppl {model_file} --require webppl/node_modules/physics --random-seed {seed}"
    print(f"Running task {task_id}:\n{cmd}")

    # Run with shell=False (NOTE(GG): Not working in parallel right now)
    # cmd = ["webppl", model_file, "--require node_modules/physics", f"--random-seed {seed}"]
    # print(f"Running task {task_id}:\n{' '.join(cmd)}")

    time_start = time.time()
    try:
        completed_process = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"WARNING: Exceeded {timeout}s timeout while running task {task_id}")
        completed_process = None

    if completed_process is not None:
        setattr(completed_process, "runtime", time.time() - time_start)
        print(f"Task {task_id}: Simulation completed in {completed_process.runtime:.3f}s")
        if completed_process.returncode == 0:
            print(completed_process.stdout)
        else:
            print(f"WARNING: WebPPL encountered error running task {task_id}:")
            print(completed_process.stderr)

    with open(results_file, "wb") as f:
        pickle.dump(completed_process, f)

    return completed_process

def run_experiment(
    experiment_id: str,
    use_cached: bool = False,
    n_simulations: int = 10,
    n_participants: int = 20,
    physical_gaussian_pos_noise: float = 2,
    physical_gaussian_size_noise: float = 1,
    timeout: int = 120,
    seed: int = 123,
    parallel: bool = False,
):
    model_dir = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, DIR_MODELS)
    results_dir = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, DIR_RESULTS)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(
        os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_CODEX, "results.csv"),
        index_col="task_id",
        keep_default_na=False,
    )

    # Write out model files
    for task_id, row in df.iterrows():
        write_model_for_task(df, task_id, n_simulations, n_participants, physical_gaussian_pos_noise, physical_gaussian_size_noise, model_dir)

    # Run simulations
    if parallel:
        pool = multiprocessing.Pool(os.cpu_count())
        print(f"Initialized multiprocessing pool with {os.cpu_count()} workers.")

        args_list = [(task_id, model_dir, results_dir) for task_id in df.index]
        kwargs = {
            "use_cached": use_cached,
            "timeout": timeout,
            "seed": seed,
        }

        job_list = [
            pool.apply_async(run_single_task, args, kwargs) for args in args_list
        ]
        completed_process_list = [job.get() for job in tqdm(job_list)]

        # Terminate multiprocessing
        pool.close()
        pool.join()
        pool.terminate()

    else:
        completed_process_list = []
        for task_id in df.index:
            completed_process = run_single_task(
                task_id,
                model_dir,
                results_dir,
                use_cached=use_cached,
                timeout=timeout,
                seed=seed,
            )
            completed_process_list.append(completed_process)

    # Parse simulation results
    results_list = []
    for i, completed_process in enumerate(completed_process_list):
        task_id = i + 1

        if completed_process is not None:
            if completed_process.returncode == 0:
                results_json = json.loads(completed_process.stdout)
                results_list.append(
                    {
                        "task_id": task_id,
                        "probs": results_json["probs"],
                        "runtime": completed_process.runtime,
                        "support": results_json["support"],
                        "stderr": None,
                    }
                )
            else:
                print(f"WARNING: No results for task {task_id} due to runtime error.")
                results_list.append(
                    {
                        "task_id": task_id,
                        "runtime": completed_process.runtime,
                        "stderr": completed_process.stderr,
                    }
                )
        else:
            print(f"WARNING: No results for task {task_id} due to timeout.")
            results_list.append(
                {
                    "task_id": task_id,
                }
            )

    df_results = pd.DataFrame(results_list)
    df_results = df_results.set_index("task_id")
    df_results.to_csv(
        os.path.join(
            DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, "simulator_results.csv"
        )
    )

    return df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--use_cached", action="store_true", default=False, help="Checks the results dir for prior results.pkl files and uses these when available.")
    parser.add_argument("--n_simulations", type=int, default=10, help="Number of physics simulations to run for each participant.")
    parser.add_argument("--n_participants", type=int, default=20, help="Number of (simulated) participants. Each participant runs `n_simulations` mental simulations and reports the average on a likert scale.")
    parser.add_argument("--physical_gaussian_pos_noise", type=float, default=2, help="How much gaussian noise is running in the physics simulation")
    parser.add_argument("--physical_gaussian_size_noise", type=float, default=2, help="How much gaussian noise is running in the physics simulation")
    parser.add_argument("--timeout", type=int, default=120, help="Per-task timeout (seconds) after which simulation will be terminated.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed to pass to WebPPL.")
    parser.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction, default=False, help="Runs the simulations in parallel. Speedup is approx. by a factor of n_cpus."
    )
    args = parser.parse_args()

    run_experiment(
        experiment_id=args.experiment_id,
        use_cached=args.use_cached,
        n_participants=args.n_participants,
        n_simulations=args.n_simulations,
        physical_gaussian_pos_noise=args.physical_gaussian_pos_noise,
        physical_gaussian_size_noise=args.physical_gaussian_size_noise,
        timeout=args.timeout,
        seed=args.seed,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()