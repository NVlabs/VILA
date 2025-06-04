import argparse
import json
import os
import re
import sys
from urllib.parse import urlparse, urlunparse

import braceexpand

from . import wids, wids_dl
from .wids_specs import load_remote_dsdesc_raw


def format_with_suffix(num):
    suffixes = ["", "k", "M", "G", "T", "E"]
    i = 0
    while num >= 1000 and i < len(suffixes) - 1:
        num /= 1000.0
        i += 1
    return f"{num:.1f}{suffixes[i]}"


class AtomicJsonUpdate:
    def __init__(self, filename):
        self.filename = filename
        self.backup_filename = filename + ".bak"
        self.temp_filename = filename + ".temp"
        self.data = None

    def __enter__(self):
        # Read the original file
        with open(self.filename) as file:
            self.data = json.load(file)
        return self.data

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # Write the modified data to the temporary file
            with open(self.temp_filename, "w") as file:
                json.dump(self.data, file, indent=2)
            # Rename the original file to a backup
            os.rename(self.filename, self.backup_filename)
            # Rename the new file to the original file name
            os.rename(self.temp_filename, self.filename)
        elif os.path.exists(self.temp_filename):
            os.remove(self.temp_filename)


def urldir(url):
    """Return the directory part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = os.path.dirname(path)
    return parsed_url._replace(path=directory).geturl()


def urlfile(url):
    """Return the file part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)
    return filename


def urldirbase(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Use 'file' scheme if no scheme is given
    scheme = parsed_url.scheme or "file"

    # Handle file URLs and relative paths
    if scheme == "file" and not parsed_url.netloc:
        path = os.path.abspath(parsed_url.path)
    else:
        path = parsed_url.path
    # Get the directory without the filename
    path_without_filename = os.path.dirname(path)

    # Reconstruct URL without filename
    url_without_filename = urlunparse((scheme, parsed_url.netloc, path_without_filename, "", "", ""))

    return url_without_filename


def shorten_name(s):
    l = re.split(r"[^a-zA-Z0-9_]+", s)
    found = set()
    result = []
    for word in l:
        if re.match(r"^[0-9]*$", word):
            continue
        if word not in found:
            result.append(word)
            found.add(word)
    return "-".join(result)


def main_create(args):
    """Create a full shard index for a list of files."""
    # set default output file name
    if args.output is None:
        args.output = "shardindex.json"

    if args.name is None:
        first = os.path.splitext(args.files[0])[0]
        args.name = shorten_name(first)
        print("setting name to", args.name)

    # read the list of files from stdin if there is only one file and it is "-"
    if len(args.files) == 1 and args.files[0] == "-":
        args.files = [line.strip() for line in sys.stdin]

    # expand any brace expressions in the file names
    fnames = []
    for f in args.files:
        fnames.extend(braceexpand.braceexpand(f))

    # create the shard index
    downloader = wids_dl.SimpleDownloader()
    files = []
    for fname in fnames:
        print(fname)
        downloaded = downloader.download(fname, "/tmp/shard.tar")
        md5sum = wids.compute_file_md5sum(downloaded)
        nsamples = wids.compute_num_samples(downloaded)
        filesize = os.stat(downloaded).st_size
        files.append(dict(url=fname, md5sum=md5sum, nsamples=nsamples, filesize=filesize))
        downloader.release(downloaded)

    files = sorted(files, key=lambda x: x["url"])

    # create the result dictionary
    result = dict(
        __kind__="wids-shard-index-v1",
        wids_version=1,
        shardlist=files,
    )

    if args.name != "":
        result["name"] = args.name

    if args.base is not None:
        result["base"] = args.base

    # add info if it is given
    if args.info is not None:
        info = open(args.info).read()
        result["info"] = info

    # write the result
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


def main_update(args):
    """Update an existing file."""
    with AtomicJsonUpdate(args.filename) as data:
        if args.name != "":
            data["name"] = args.name
        if args.keep:
            data["keep"] = True
        if args.nokeep:
            data["keep"] = False
        if args.info != "":
            data["info"] = args.info
        if args.base != "":
            data["base"] = args.base
        if args.rebase:
            bases = {urldirbase(shard["url"]) for shard in data["shardlist"]}
            assert len(bases) == 1, f"multiple/no bases found: {bases}"
            base = bases.pop()
            print(f"rebasing to {base}")
            data["base"] = base
        if args.dir != "" or args.nodir or args.rebase:
            shardlist = data["shardlist"]
            for shard in shardlist:
                url = shard["url"]
                file = urlfile(url)
                if args.nodir:
                    shard["url"] = file
                else:
                    shard["url"] = os.path.join(args.dir, file)
            data["shardlist"] = sorted(shardlist, key=lambda x: x["url"])
        if "name" not in data:
            parsed = urlparse(args.filename)
            data["name"] = os.path.splitext(os.path.basename(parsed.path))[0]


def print_long_info(data, filename):
    print("            name:", data.get("name"))
    print("            info:", data.get("info"))
    print("            base:", data.get("base"), f"(assumed: {urldir(filename)})")
    total_size = sum(shard["filesize"] for shard in data["shardlist"])
    total_samples = sum(shard["nsamples"] for shard in data["shardlist"])
    print("      total size:", format_with_suffix(total_size))
    print("   total samples:", format_with_suffix(total_samples))
    print(" avg sample size:", format_with_suffix(int(total_size / total_samples)))
    print(
        "  avg shard size:",
        format_with_suffix(int(total_size / len(data["shardlist"]))),
    )
    print("     first shard:", data["shardlist"][0]["url"])
    print("      last shard:", data["shardlist"][-1]["url"])
    if len(data.get("datasets", [])) > 0:
        print("        datasets:")
        for dataset in data.get("datasets", []):
            print("    dataset name:", dataset.get("name"))
            print(
                "     dataset url:",
                dataset.get("source_url", len(data.get("shardlist", []))),
            )


def main_info(args):
    """Show info about an index file."""
    if args.table:
        print("file\tname\tnbytes\tnsamples\tbase\tlast\tdatasets")
        for filename in args.filenames:
            data = load_remote_dsdesc_raw(filename)
            print(
                filename,
                data.get("name"),
                sum(shard["filesize"] for shard in data["shardlist"]),
                sum(shard["nsamples"] for shard in data["shardlist"]),
                data.get("base"),
                data["shardlist"][-1]["url"],
                len(data.get("datasets", [])),
                sep="\t",
            )
    else:
        for filename in args.filenames:
            data = load_remote_dsdesc_raw(filename)
            print("filename:", filename)
            print_long_info(data, filename)
            print()


def maybe_read(x):
    try:
        return x.read()
    except AttributeError:
        return x


def maybe_decode(sample):
    sample = {k: maybe_read(v) for k, v in sample.items()}
    return sample


def main_sample(args):
    raw = args.raw or args.cat is not None
    if raw:
        ds = wids.ShardListDataset(args.filename, transformations=[maybe_decode])
    else:
        ds = wids.ShardListDataset(args.filename)
    print("dataset size:", len(ds), file=sys.stderr)
    sample = ds[args.index]
    if args.cat is not None:
        sys.stdout.buffer.write(sample[args.cat])
        return 0
    mkl = max(len(k) for k in sample.keys())
    for k, v in sorted(sample.items()):
        print(k.ljust(mkl), repr(v)[: args.width - mkl - 1])


def main():
    """Commands for manipulating the shard index."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Command line tool with subcommands for file operations.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Create the parser for the "create" command
    create_parser = subparsers.add_parser("create", help="Create a new file")
    create_parser.add_argument("files", nargs="+", help="files to index")
    create_parser.add_argument("--output", "-o", help="output file name")
    create_parser.add_argument("--name", "-n", help="name for dataset", default=None)
    create_parser.add_argument("--info", "-i", help="description for dataset", default=None)
    create_parser.add_argument("--base", "-b", help="base path", default=None)

    # Create the parser for the "update" command
    update_parser = subparsers.add_parser("update", help="Update an existing file")
    update_parser.add_argument("filename", type=str, help="Name of the file to update")
    update_parser.add_argument("-n", "--name", default="", help="set the dataset name")
    update_parser.add_argument("-k", "--keep", default="store_true", help="set the keep flag")
    update_parser.add_argument("-K", "--nokeep", default="store_true", help="clear the keep flag")
    update_parser.add_argument("-i", "--info", default="", help="set the dataset info")
    update_parser.add_argument("-D", "--nodir", action="store_true", help="remove the directory from the URLs")
    update_parser.add_argument("-d", "--dir", default="", help="set the directory on the URLs")
    update_parser.add_argument("-b", "--base", default="", help="set the base")
    update_parser.add_argument("-B", "--rebase", action="store_true", help="rebase the URLs")

    # Create the parser for the "info" command
    info_parser = subparsers.add_parser("info", help="Show info about an index file")
    info_parser.add_argument("filenames", type=str, nargs="*", help="Name of the file to display")
    info_parser.add_argument("-t", "--table", action="store_true", help="output in table format")

    # Create the parser for the "sample" command
    sample_parser = subparsers.add_parser("sample", help="Show info about an index file")
    sample_parser.add_argument("filename", type=str, help="Name of the file to update")
    sample_parser.add_argument("index", type=int, default=0, help="Index of the sample to show")
    sample_parser.add_argument("-p", "--python", action="store_true", help="Show raw sample")
    sample_parser.add_argument("-r", "--raw", action="store_true", help="Show raw sample")
    sample_parser.add_argument("-c", "--cat", type=str, default=None, help="Output the bytes for a given key")
    sample_parser.add_argument("-w", "--width", type=int, default=250, help="Output the bytes for a given key")

    # Parse the arguments
    args = parser.parse_args()  # Dynamically call the appropriate function based on the subcommand

    try:
        func = getattr(sys.modules[__name__], f"main_{args.command}")
    except AttributeError:
        parser.print_help()

    func(args)


if __name__ == "__main__":
    main()
