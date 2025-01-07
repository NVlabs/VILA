from dataclasses import dataclass

from transformers import PretrainedConfig


@dataclass
class QuantizationConfig:
    quantize_model: str = "false"
    symm: bool = True
    epsilon: float = 1e-10
    fabit: str = "E4M3"
    fwbit: str = "E4M3"
    bobit: str = "E5M2"
    row_blocksize: int = -1
    col_blocksize: int = -1
    qchoice: str = "none"
    pad_to_multiple_of: int = 0

    def __init__(
        self,
        quantize_model,
        symm,
        epsilon,
        fabit,
        fwbit,
        bobit,
        row_blocksize,
        col_blocksize,
        qchoice,
        pad_to_multiple_of,
        **kwargs,
    ):
        super().__init__()
        self.quantize_model = quantize_model
        self.symm = symm
        self.epsilon = epsilon
        self.fabit = fabit
        self.fwbit = fwbit
        self.bobit = bobit
        self.row_blocksize = row_blocksize
        self.col_blocksize = col_blocksize
        self.qchoice = qchoice
        self.pad_to_multiple_of = pad_to_multiple_of


# class QuantizationConfig(PretrainedConfig):
#     def __init__(
#         self,
#         quantize_model="false",
#         symm=True,
#         epsilon=1e-10,
#         fabit="E4M3",
#         fwbit="E4M3",
#         bobit="E5M2",
#         row_blocksize=-1,
#         col_blocksize=-1,
#         qchoice="none",
#         pad_to_multiple_of=0,
#         **kwargs,
#     ):
#         super().__init__()
#         self.quantize_model = quantize_model
#         self.symm = symm
#         self.epsilon = epsilon
#         self.fabit = fabit
#         self.fwbit = fwbit
#         self.bobit = bobit
#         self.row_blocksize = row_blocksize
#         self.col_blocksize = col_blocksize
#         self.qchoice = qchoice
#         self.pad_to_multiple_of = pad_to_multiple_of


