""" AutoResume callback.

A transformer trainer callback for interfacing with ADLR's AutoResume SDK.

Copyright 2024 NVIDIA CORPORATION.
"""
import os
import sys

import torch
import transformers
from transformers.utils import logging

logger = logging.get_logger("transformers")


def rank_print(*s):
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    print(rank, *s)


sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
try:
    logger.info("Importing AutoResume lib...")
    from userlib.auto_resume import AutoResume

    AutoResume.init()
    logger.info("Found AutoResume SDK!")
except:
    logger.warn("Did not find AutoResume SDK!")
    AutoResume = None


class AutoResumeCallback(transformers.TrainerCallback):
    """
    A [`TrainerCallback`] that handles autoresume.

    Args:
        interval: interval (in number of iterations) between checks as to
            whether to suspend.
    """

    def __init__(self, interval: int = 50):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            rank_print("AutoResumeHook: Checking whether to suspend...")

            # Check whether to suspend the job.
            should_preempt = AutoResume is not None and AutoResume.termination_requested()

            if should_preempt:
                if state.is_local_process_zero:
                    logger.warn(f"AutoResumeHook: Request resume...")
                    if AutoResume is not None:
                        AutoResume.request_resume()
                control.should_training_stop = True
                control.should_save = True
