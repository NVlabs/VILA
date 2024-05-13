import transformers
from transformers import AutoTokenizer
from llava.data.dataset import preprocess_v1
import torch
import unittest


class TestTokenizerWarning(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, legacy=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.test_conv = [
            [
                {
                    "from": "human",
                    "value": "<image> is the weather sunny ",
                },
                {
                    "from": "gpt",
                    "value": "y ",
                },
                {
                    "from": "human",
                    "value": "can you see a fence ",
                },
                {
                    "from": "gpt",
                    "value": "yes ",
                },
                {
                    "from": "human",
                    "value": "what is it made of ",
                },
                {
                    "from": "gpt",
                    "value": "wire ",
                },
            ]
        ]
        self.expected_tensor = (
            torch.tensor(
                [
                    [
                        1,
                        319,
                        13563,
                        1546,
                        263,
                        12758,
                        1404,
                        322,
                        385,
                        23116,
                        21082,
                        20255,
                        29889,
                        450,
                        20255,
                        4076,
                        8444,
                        29892,
                        13173,
                        29892,
                        322,
                        1248,
                        568,
                        6089,
                        304,
                        278,
                        1404,
                        29915,
                        29879,
                        5155,
                        29889,
                        3148,
                        1001,
                        29901,
                        29871,
                        -200,
                        29871,
                        338,
                        278,
                        14826,
                        6575,
                        1460,
                        29871,
                        319,
                        1799,
                        9047,
                        13566,
                        29901,
                        343,
                        29871,
                        2,
                        11889,
                        29901,
                        508,
                        366,
                        1074,
                        263,
                        285,
                        663,
                        29871,
                        319,
                        1799,
                        9047,
                        13566,
                        29901,
                        4874,
                        29871,
                        2,
                        11889,
                        29901,
                        825,
                        338,
                        372,
                        1754,
                        310,
                        29871,
                        319,
                        1799,
                        9047,
                        13566,
                        29901,
                        8014,
                        29871,
                        2,
                    ]
                ]
            )
            .int()
        )

    def test_token_ids_equal(self):

        processed_dict = preprocess_v1(self.test_conv, tokenizer=self.tokenizer, has_image=True)
        tokenized_tensor = processed_dict["input_ids"]
        self.assertTrue(torch.sum(tokenized_tensor == self.expected_tensor).item() == tokenized_tensor.numel())


if __name__ == "__main__":
    unittest.main()
