import argparse
import datetime
import os
import os.path as osp
import subprocess

from termcolor import colored


def supports_gpus_per_node():
    VILA_DATASETS = os.environ.get("VILA_DATASETS", "")
    if "eos" in VILA_DATASETS.lower():
        return False
    return True


def run_inside_slurm_node():
    if os.environ.get("NO_SLURM"):
        print(colored("Env NO_SLURM has been set. Ignore slurm-related args.", "yellow"))
        return True
    if os.environ.get("SLURM_JOB_ID"):
        print(colored("Running inside slurm node. Ignore slurm-related args.", "yellow"))
        return True
    if subprocess.run("which srun", shell=True).returncode != 0:
        print(colored("No slurm installed. Ignore slurm-related args.", "yellow"))
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", "-J", type=str, required=True)
    parser.add_argument("--nodes", "-N", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--mode", "-m", type=str, default="train")
    parser.add_argument("--time", "-t", type=str, default="4:00:00")
    parser.add_argument("--timedelta", type=int, default=5)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--max-retry", type=int, default=-1)
    # -1: indicates none, for train jobs, it will be set 3 and otherwise 1
    parser.add_argument("--pty", action="store_true")
    parser.add_argument("--uuid", action="store_true")
    parser.add_argument("--email", type=str)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.max_retry < 0:
        if args.mode == "train":
            args.max_retry = 3
        else:
            args.max_retry = 0

    # Generate run name and output directory
    if "%t" in args.job_name:
        args.job_name = args.job_name.replace("%t", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if args.output_dir is None:
        args.output_dir = os.path.join("runs", args.mode, args.job_name)
    output_dir = os.path.expanduser(args.output_dir)

    # Calculate the timeout
    time = datetime.datetime.strptime(args.time, "%H:%M:%S")
    if time < datetime.datetime.strptime("0:01:00", "%H:%M:%S"):
        raise ValueError("Time must be at least 1 minutes")
    timeout = time - datetime.timedelta(minutes=args.timedelta)
    timeout = timeout.hour * 60 + timeout.minute
    timeout = f"{timeout}m"

    # Set environment variables
    env = os.environ.copy()
    env["RUN_NAME"] = args.job_name
    env["OUTPUT_DIR"] = output_dir

    if run_inside_slurm_node():
        full_cmd = " ".join(args.cmd)
    else:
        # Get SLURM account and partition
        if "VILA_SLURM_ACCOUNT" not in os.environ or "VILA_SLURM_PARTITION" not in os.environ:
            raise ValueError("`VILA_SLURM_ACCOUNT` and `VILA_SLURM_PARTITION` must be set in the environment.")
        account = os.environ["VILA_SLURM_ACCOUNT"]
        partition = os.environ["VILA_SLURM_PARTITION"]

        # Compose the SLURM command
        job_name = f"{account}:{args.mode}/{args.job_name}"
        cmd = ["srun"]
        cmd += ["--account", account]
        cmd += ["--partition", partition]
        cmd += ["--job-name", job_name]

        # Send email on the job status if specified
        if args.email:
            cmd += ["--mail-type", "BEGIN,END,FAIL"]
            cmd += ["--mail-user", args.email]

        if not args.pty:
            # Redirect output to files if not pty / interactive
            cmd += ["--output", f"{output_dir}/slurm/%J.out"]
            cmd += ["--error", f"{output_dir}/slurm/%J.err"]
        cmd += ["--nodes", str(args.nodes)]
        if supports_gpus_per_node() and args.gpus_per_node > 0:
            # eos slurm does not support gpus-per-node option
            cmd += ["--gpus-per-node", str(args.gpus_per_node)]
        if args.uuid:
            cmd += ["--dependency", "singleton"]
        cmd += ["--time", args.time]
        cmd += ["--exclusive"]
        cmd += ["timeout", timeout]
        cmd += args.cmd
        full_cmd = " ".join(cmd)
    print(colored(full_cmd, attrs=["bold"]))

    # Run the job and resume if it times out
    fail_times = 0
    while True:
        returncode = subprocess.run(full_cmd, env=env, shell=True).returncode
        print(f"returncode: {returncode}")
        if returncode == 0:
            print("Job finished successfully!")
            break
        if returncode != 124:
            fail_times += 1
            if fail_times > args.max_retry:
                break
            print(f"Job failed, retrying {fail_times} / {args.max_retry}")
        else:
            fail_times = 0
            print(f"Job timed out, retrying...")

    # Exit with the return code
    print(f"Job finished with exit code {returncode}")
    exit(returncode)


if __name__ == "__main__":
    main()


