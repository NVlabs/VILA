#include <torch/extension.h>

#include "include/fp8_adamw.h"
#include "include/fp8_adamw_expand.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_adamw_step", &FP8_AdamW, "Update the quantized optimizer states");
  m.def("fp8_adamw_expand_step", &FP8_AdamW_expand,
        "Update the quantized optimizer states, use polynomial expander");
}
