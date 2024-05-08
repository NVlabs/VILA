#!/bin/bash

# Define the file to be modified
FILE_PATH="/usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/llama/convert.py"

# Backup the original file before modification
cp $FILE_PATH "${FILE_PATH}.bak"

# Replace the strings
sed -i ':a;N;$!ba;s|hf_config = LlavaConfig.from_pretrained(hf_model).text_config|hf_config = LlavaConfig.from_pretrained(hf_model).text_config\n    if hf_config.model_type == "llava_llama":\n        hf_config.llm_cfg["architecture"] = hf_config.llm_cfg["architectures"]\n        hf_config.llm_cfg["dtype"] = hf_config.llm_cfg["torch_dtype"]\n        hf_config = PretrainedConfig.from_dict(hf_config.llm_cfg)|g' $FILE_PATH
sed -i ':a;N;$!ba;s|if "vila" in model_dir:\n        sys.path.append(model_dir + "/../VILA")\n        from llava.model import LlavaConfig, LlavaLlamaForCausalLM\n        AutoConfig.register("llava_llama", LlavaConfig)\n        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)|# if "vila" in model_dir:\n#     sys.path.append(model_dir + "/../VILA")\n#     from llava.model import LlavaConfig, LlavaLlamaForCausalLM\n#     AutoConfig.register("llava_llama", LlavaConfig)\n#     AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)|g' $FILE_PATH

# Inform the user
echo "Replacement done. Original file backed up as ${FILE_PATH}.bak"
