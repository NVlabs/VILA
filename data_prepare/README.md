# Data Preparation for Training VILA

To train VILA, we used the following datasets:

| Stage                   | Datasets                                                                         |
| ----------------------- | -------------------------------------------------------------------------------- |
| 1. Initialize projector | CC3M                                                                             |
| 2. Pre-training         | MMC4-core, COYO-700M, ShreGPT4V_pretrain                                                      |
| 3. SFT                  | LLaVA-Next mixture, VFLAN, WIT, GSM8K-ScRel-SFT, Sherlock, ScienceQA, Shot2story, Video_ChatGPT, Youcook2, Vatex, ShareGPT_Video |

### LLaVa-CC3M-Pretrain

We use [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) to train the visual language projector

### MMC4-Core Dataset

Due to the limit of compute, we pre-train VILA on the smaller core set of MMC4 instead of the full set.

1. Firstly, download the annotations of the MMC4-core dataset here: https://github.com/allenai/mmc4. We used the non-fewer-face split, and you may need to request the access [here](https://forms.gle/VYtcNY8aYaUANK9f8).

1. Now modify the input and output path in `mmc4_downloader.py` and run the following script to scrawl the MMC4 images:

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

### LLaVA-1.5 Instruction Data

We use this [file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) in our experiments. Please download this dataset from LLaVA authors.

```bash
huggingface-cli download liuhaotian/LLaVA-Instruct-150K llava_v1_5_mix665k.json --repo-type dataset
```

### VFlan dataset

#### TextFLAN

1. Download FLAN datasets:

```bash
huggingface-cli download Open-Orca/FLAN --repo-type dataset --local-dir FLAN --local-dir-use-symlinks False
```

2. Preprocess FLAN dataset (sample 1M data from 378M samples):

```bash
cd sft
python preprocess_flan.py
```

#### M3IT Dataset

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

### LLaVA-Next mixture

You can follow this [page](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) to prepare the data mixture that is proposed by LLaVA-Next.

### Shot2story

Please follow this [page](https://github.com/bytedance/Shot2Story/blob/master/DATA.md) to download the videos. The JSON file can be downloaded with

```bash
huggingface-cli download mit-han-lab/vila-dataset shot2story_shotonly.json
 --repo-type dataset --local-dir shot2story --local-dir-use-symlinks False
```

### Video_ChatGPT

You can follow this [page](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/README.md#video-instruction-dataset-open_file_folder) to prepare Video_ChatGPT dataset.

### Youcook2

Please follow this [page](http://youcook2.eecs.umich.edu/) to download the videos. The JSON file can be downloaded with

```bash
huggingface-cli download mit-han-lab/vila-dataset youcook_filtered_v3.json --repo-type dataset --local-dir youcook2 --local-dir-use-symlinks False
```

### Vatex

Please follow this [page](https://eric-xw.github.io/vatex-website/download.html) to download the videos. The JSON file can be downloaded with

```bash
huggingface-cli download mit-han-lab/vila-dataset vatex_filtered_v3.json --repo-type dataset --local-dir vatex --local-dir-use-symlinks False
```

### ShareGPT_Video

You can follow this [page](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction) to prepare ShareGPT_Video dataset.

### WIT

The original WIT data can be obtained [google-research-datasets/wit](https://github.com/google-research-datasets/wit/tree/main). * We subsample ~538K english data from the original WIT dataset and curate a llava conversation format JSON file.

```bash
huggingface-cli download mit-han-lab/vila-dataset wit_processed_538k.json --repo-type dataset --local-dir WIT --local-dir-use-symlinks False
```

### GSM8K-ScRel-SFT

We add some math data [gsm8k-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl) to our SFT stage.

### Sherlock

The image files of Sherlock can be obtained from [VisualGenome](https://visualgenome.org/api/v0/api_home.html) and [VCR](https://visualcommonsense.com/download/) separately. The llava conversation format JSON file can be downloaded with

```bash
huggingface-cli download mit-han-lab/vila-dataset sherlock_317k.json --repo-type dataset --local-dir sherlock --local-dir-use-symlinks False
```

### ScienceQA

We use the train split of ScienceQA. The image data of the train split can be obtained from [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) or their [huggingface repo](https://huggingface.co/datasets/derek-thomas/ScienceQA). The llava conversation format JSON file can be downloaded with

```bash
huggingface-cli download mit-han-lab/vila-dataset scienceqa_train_12k.json --repo-type dataset --local-dir scienceqa --local-dir-use-symlinks False
```
