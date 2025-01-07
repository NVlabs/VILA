"""
# dynamically create a shard index
class DirectoryDataset(ShardListDataset):
    def __init__(self, directory):
        pass
"""

"""
# randomly choose shards from a directory
class DirectoryQueueDataset(IterableDataset):
    def __init__(self, directory, strategy="replace", choice="random", downloader=None, transformations="PIL"):
        pass
    def add_transform(self, transform):
        pass
    def __iter__(self):
        # pick file according to strategy
        # rename file to .active
        # randomly yield samples from file
        # rename file back to its original name or unlink it, according to strategy
        pass
"""


