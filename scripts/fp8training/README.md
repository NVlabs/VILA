# FP8 Training for VILA

VILA's fp8 training are powered by COAT. To enable, adding following args to training

```diff
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
-        --report_to wandb
+        --report_to wandb \
+        --quantize_model "fp8Linear_qwen2" \
+        --fabit "E4M3" \
+        --fwbit "E4M3" \
+        --bobit "E5M2" \
+        --row_blocksize -1 \
+        --col_blocksize -1 \
+        --pad_to_multiple_of 128
```

We provide examples of VILA [training with FP16](./sft_qwen_fp16.sh) and [training with FP8](./sft_qwen_fp8.sh) to reproduce.
