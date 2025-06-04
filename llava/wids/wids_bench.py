import argparse
import json

from . import wids
from .compat import WebDataset


def main_wids(args):
    desc = json.load(open(args.dataset))
    files = desc["files"]
    dataset = wids.ShardListDataset(files, cache_size=4)
    print(len(dataset))
    for i in range(len(dataset)):
        print(i, dataset[i]["__key__"])
    dataset.close()


def main_wds(args):
    desc = json.load(open(args.dataset))
    files = desc["files"]
    urls = [f["url"] for f in files]
    dataset = WebDataset(urls)
    for i, sample in enumerate(dataset):
        print(i, sample["__key__"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # there are two subcommands: wids and wds
    subparsers = parser.add_subparsers(dest="command")
    wids_parser = subparsers.add_parser("wids")
    wds_parser = subparsers.add_parser("wds")

    # wids subcommand
    wids_parser.add_argument("dataset", help="dataset name")

    # wds subcommand
    wds_parser.add_argument("dataset", help="dataset name")

    args = parser.parse_args()

    if args.command == "wids":
        main_wids(args)
    elif args.command == "wds":
        main_wds(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
