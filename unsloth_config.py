HAS_XPU = True
HAS_BNB = False
HAS_XFORMERS = False
ENABLE_BENCHMARK = True

device_name = "xpu" if HAS_XPU else "cuda"
device_id = "xpu:0" if HAS_XPU else "cuda:0"

causal_mask_type = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else bool

