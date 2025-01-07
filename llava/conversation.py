# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

import dataclasses
from enum import Enum, auto
from typing import List

from llava.utils.logging import logger


class SeparatorStyle(Enum):
    """Different separator style."""

    AUTO = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    sep_style: SeparatorStyle = SeparatorStyle.AUTO
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep
            for rid, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    sep = self.sep if rid < len(messages) - 1 else self.sep2
                    ret += role + message + sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


conv_auto = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.AUTO,
    sep="\n",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

hermes_2 = Conversation(
    system="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
    messages=(),
    version="hermes-2",
)

# Template added by Yukang. Note (kentang-mit@): sep is <|eot_id|> for official template.
llama_3_chat = Conversation(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
    sep2="<|end_of_text|>",
)


default_conversation = conv_auto
conv_templates = {
    "auto": conv_auto,
    "hermes-2": hermes_2,
    "llama_3": llama_3_chat,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "plain": conv_llava_plain,
}


CONVERSATION_MODE_MAPPING = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-8b": "llama_3",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-40b": "hermes-2",
    "llama-3": "llama_3",
    "llama3": "llama_3",
}


def auto_set_conversation_mode(model_name_or_path: str) -> str:
    global default_conversation
    for k, v in CONVERSATION_MODE_MAPPING.items():
        if k in model_name_or_path.lower():
            logger.info(f"Setting conversation mode to `{v}` based on model name/path `{model_name_or_path}`.")
            default_conversation = conv_templates[v]
            return


