class QuantizationConfig:
    def __init__(self):
        self.qlinear_config = {
            "mlp_gate": {"all", "linear"},
            "mlp_up": {"all", "linear"},
            "mlp_down": {"all", "linear"},
            "attn_proj": {"all", "linear"},
            "attn_q": {"all", "linear"},
            "attn_k": {"all", "linear"},
            "attn_v": {"all", "linear"},
        }
        self.qact_config = {
            "mul_act_in1": {"all", "gelu"},
            "mul_act_in2": {"all", "gelu", "te_like"},
            "mul_act_out": {"all", "gelu", "te_like"},
            "mlp_act_sum": {"all", "mlp", "te_like"},
            "mlp_act_gate": {"all", "mlp", "te_like"},
            "mlp_act_up": {"all", "mlp", "te_like"},
            "mlp_act_in": {"all", "mlp", "te_like"},
            "mlp_act_out": {"all", "mlp"},
            "ln_attn_in": {"all", "layernorm"},
            "ln_mlp_in": {"all", "layernorm"},
            "ln_attn_out": {"all", "layernorm", "te_like"},
            "ln_mlp_out": {"all", "layernorm", "te_like"},
            "add_attn_in_re": {"all", "residual"},
            "add_attn_in_fx": {"all", "residual", "te_like"},
            "add_mlp_in_re": {"all", "residual"},
            "add_mlp_in_fx": {"all", "residual", "te_like"},
            "re_attn_out_re": {"all", "residual"},
            "re_attn_out_fx": {"all", "residual"},
            "re_mlp_out_re": {"all", "residual"},
            "re_mlp_out_fx": {"all", "residual"},
            "attn_qkv_sum": {"all", "attn", "te_like"},
            "attn_q_in": {"all", "attn", "te_like"},
            "attn_k_in": {"all", "attn", "te_like"},
            "attn_v_in": {"all", "attn", "te_like"},
            "attn_q_out": {"all", "attn", "te_like"},
            "attn_k_out": {"all", "attn", "te_like"},
            "attn_v_out": {"all", "attn", "te_like"},
            "attn_proj_in": {"all", "attn", "te_like"},
        }

        self.qgelu_config = {"mlp_gelu": {"all", "gelu"}}

        self.qlayernorm_config = {"ln_attn": {"all", "layernorm"}, "ln_mlp": {"all", "layernorm"}}

        self.qadd_config = {"add_attn": {"all", "residual"}, "add_mlp": {"all", "residual"}}

        self.qmul_config = {
            "mul_act": {"all", "gelu"},
        }


qconfig = QuantizationConfig()


