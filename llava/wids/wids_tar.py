import io
import os
import os.path
import pickle
import re
import tarfile

import numpy as np


def find_index_file(file):
    prefix, last_ext = os.path.splitext(file)
    if re.match("._[0-9]+_$", last_ext):
        return prefix + ".index"
    else:
        return file + ".index"


class TarFileReader:
    def __init__(self, file, index_file=find_index_file, verbose=True):
        self.verbose = verbose
        if callable(index_file):
            index_file = index_file(file)
        self.index_file = index_file

        # Open the tar file and keep it open
        if isinstance(file, str):
            self.tar_file = tarfile.open(file, "r")
        else:
            self.tar_file = tarfile.open(fileobj=file, mode="r")

        # Create the index
        self._create_tar_index()

    def _create_tar_index(self):
        if self.index_file is not None and os.path.exists(self.index_file):
            if self.verbose:
                print("Loading tar index from", self.index_file)
            with open(self.index_file, "rb") as stream:
                self.fnames, self.index = pickle.load(stream)
            return
        # Create an empty list for the index
        self.fnames = []
        self.index = []

        if self.verbose:
            print("Creating tar index for", self.tar_file.name, "at", self.index_file)
        # Iterate over the members of the tar file
        for member in self.tar_file:
            # If the member is a file, add it to the index
            if member.isfile():
                # Get the file's offset
                offset = self.tar_file.fileobj.tell()
                self.fnames.append(member.name)
                self.index.append([offset, member.size])
        if self.verbose:
            print("Done creating tar index for", self.tar_file.name, "at", self.index_file)
        self.index = np.array(self.index)
        if self.index_file is not None:
            if os.path.exists(self.index_file + ".temp"):
                os.unlink(self.index_file + ".temp")
            with open(self.index_file + ".temp", "wb") as stream:
                pickle.dump((self.fnames, self.index), stream)
            os.rename(self.index_file + ".temp", self.index_file)

    def names(self):
        return self.fnames

    def __len__(self):
        return len(self.index)

    def get_file(self, i):
        name = self.fnames[i]
        offset, size = self.index[i]
        self.tar_file.fileobj.seek(offset)
        file_bytes = self.tar_file.fileobj.read(size)
        return name, io.BytesIO(file_bytes)

    def close(self):
        # Close the tar file
        self.tar_file.close()
