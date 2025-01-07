from dataclasses import dataclass

from transformers import PretrainedConfig


@dataclass
class QuantizationConfig:
    quantize_model: str = "false"
    symm: bool = True
    epsilon: float = 1e-10
    fabit: str = "E4M3"
    fwbit: str = "E4M3"
    fobit: str = "E4M3"
    babit: str = "E5M2"
    bwbit: str = "E5M2"
    bobit: str = "E5M2"
    qchoice: str = "none"
    group_size: int = -1
    pad_to_multiple_of: int = 0
    weight_memory_efficient: bool = True

    # Legacy
    row_blocksize: int = -1
    col_blocksize: int = -1

    def __init__(
        self,
        quantize_model: str = "false",
        symm: bool = True,
        epsilon: float = 1e-10,
        fabit: str = "E4M3",
        fwbit: str = "E4M3",
        fobit: str = "E4M3",
        babit: str = "E5M2",
        bwbit: str = "E5M2",
        bobit: str = "E5M2",
        qchoice: str = "none",
        group_size: int = -1,
        pad_to_multiple_of: int = 0,
        weight_memory_efficient: bool = True,
        row_blocksize: int = -1,
        col_blocksize: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.quantize_model = quantize_model
        self.symm = symm
        self.epsilon = epsilon
        self.fabit = fabit
        self.fwbit = fwbit
        self.fobit = fobit
        self.babit = babit
        self.bwbit = bwbit
        self.bobit = bobit
        self.qchoice = qchoice
        self.group_size = group_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.weight_memory_efficient = weight_memory_efficient

        self.row_blocksize = row_blocksize
        self.col_blocksize = col_blocksize


