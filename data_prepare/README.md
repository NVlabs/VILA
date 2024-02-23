
# Data Preparation for Training VILA

To train VILA, we used the following datasets:

| Stage                   | Datasets                    |
| ----------------------- | --------------------------- |
| 1. Initialize projector | CC3M                         |
| 2. Pre-training         | MMC4-core, COYO-700M subset |
| 3. SFT                  | LLaVA-1.5, VFLAN, ShareGPT, TextFLAN      |

### LLaVa-CC3M-Pretrain
We use [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) to train the visual language projector


### MMC4-Core Dataset
Due to the limit of compute, we pre-train VILA on the smaller core set of MMC4 instead of the full set. 

1. Firstly, download the annotations of the MMC4-core dataset here: https://github.com/allenai/mmc4. We used the non-fewer-face split, and you may need to request the access [here](https://forms.gle/VYtcNY8aYaUANK9f8). 

2. Now modify the input and output path in `mmc4_downloader.py` and run the following script to scrawl the MMC4 images:
```bash
cd mmc4
python mmc4_downloader.py
```
Note that due to the expiration of image urls, you may end up getting a subset of the entire corpus. 

The scrawling may take a long time. Optionally, you can also shard the workload over multiple jobs/machines concurrently to speed up the process:

```bash
# provide the start and end index of the jsonl shard. There are 23098 - 14 shards totally
# python mmc4_downloader.py <start_idx> <end_idx>
python mmc4_downloader.py 0 1000  # worker 1
python mmc4_downloader.py 1000 2000  # worker 2
```

3. Filter out invalid samples in MMC4:

```bash
python mmc4_filter_and_counter.py
```

4. Merge images and text into a unified pickle file for each shard:

```bash
python mmc4_merger.py
```

### COYO-700M Dataset
1. Download the metadata of COYO-700M:
```bash
huggingface-cli download kakaobrain/coyo-700m --repo-type dataset --local-dir coyo-700m --local-dir-use-symlinks False
```

2. Scrawl the COYO images. Note that here we only keep a 20% subset in each shard with the highest CLIP similarity, to balance compute budget and data quality. 

There are totally 128 shards of annotations. Now download each one with the script:
```bash
cd coyo
for SHARD in {0..127}; do
    python coyo_downloader.py $SHARD  
done
```

3. Split downloaded COYO data into multiple shards:
```bash
python coyo_splitter.py
```

###  LLaVA-1.5 Instruction Data

We use this [file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) in our experiments. Please download this dataset from LLaVA authors.

```bash
huggingface-cli download liuhaotian/LLaVA-Instruct-150K llava_v1_5_mix665k.json --repo-type dataset
```


### VFlan dataset
1. Download FLAN  datasets:

```bash
huggingface-cli download Open-Orca/FLAN --repo-type dataset --local-dir FLAN --local-dir-use-symlinks False
```

2. Preprocess FLAN dataset (sample 1M data from 378M samples):

```bash
cd sft
python preprocess_flan.py
```

### M3IT Dataset
1. Download M3IT datasets:

```bash
huggingface-cli download MMInstruction/M3IT --repo-type dataset --local-dir M3IT --local-dir-use-symlinks False
```

2. Preprocess M3IT dataset:

```bash
python preprocess_m3it.py
```

3. (Optional) Split FLAN+M3IT into multiple chunks to reduce CPU memory pressure during training:

```bash
python split_vflan.py
```

### ShareGPT4v

The ShareGPT data can be obtained [mit-han-lab/ShareGPT4V](https://huggingface.co/datasets/mit-han-lab/ShareGPT4V).
    * Note the original ShareGPT4v dataset contains some samples with file ids (sa_XXXX) and repeative response. We filter those bad examples and reduced the samples from 100K -> 96K (for caption) and 1.2m -> 1.17m (for pretraining). Then we re-combine them into a single file.

```bash
huggingface-cli download mit-han-lab/ShareGPT4V --repo-type dataset --local-dir coyo-700m --local-dir-use-symlinks False
```
