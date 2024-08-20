import argparse
import datetime
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", "-J", type=str, required=True)
    parser.add_argument("--nodes", "-N", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--mode", "-m", type=str, default="train")
    parser.add_argument("--time", "-t", type=str, default="4:00:00")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Generate run name and output directory
    if "%t" in args.job_name:
        args.job_name = args.job_name.replace("%t", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    output_dir = os.path.join("runs", args.mode, args.job_name)

    # Calculate the timeout
    time = datetime.datetime.strptime(args.time, "%H:%M:%S")
    if time < datetime.datetime.strptime("0:05:00", "%H:%M:%S"):
        raise ValueError("Time must be at least 5 minutes")
    timeout = time - datetime.timedelta(minutes=5)
    timeout = timeout.hour * 60 + timeout.minute
    timeout = f"{timeout}m"

    # Get SLURM account and partition
    if "VILA_SLURM_ACCOUNT" not in os.environ or "VILA_SLURM_PARTITION" not in os.environ:
        raise ValueError("`VILA_SLURM_ACCOUNT` and `VILA_SLURM_PARTITION` must be set in the environment.")
    account = os.environ["VILA_SLURM_ACCOUNT"]
    partition = os.environ["VILA_SLURM_PARTITION"]

    # Set environment variables
    env = os.environ.copy()
    env["RUN_NAME"] = args.job_name
    env["OUTPUT_DIR"] = output_dir

    # Compose the SLURM command
    cmd = ["srun"]
    cmd += ["--account", account]
    cmd += ["--partition", partition]
    cmd += ["--job-name", f"{account}:{args.mode}/{args.job_name}"]
    cmd += ["--output", f"{output_dir}/slurm/%J-%t.out"]
    cmd += ["--error", f"{output_dir}/slurm/%J-%t.err"]
    cmd += ["--nodes", str(args.nodes)]
    cmd += ["--gpus-per-node", str(args.gpus_per_node)]
    cmd += ["--time", args.time]
    cmd += ["--exclusive"]
    cmd += ["timeout", timeout]
    cmd += args.cmd
    print(" ".join(cmd))

    # Run the job and resume if it times out
    while True:
        returncode = subprocess.run(cmd, env=env).returncode
        if returncode != 124:
            break
        print("Job timed out, retrying...")
    print(f"Job finished with exit code {returncode}")


if __name__ == "__main__":
    main()
