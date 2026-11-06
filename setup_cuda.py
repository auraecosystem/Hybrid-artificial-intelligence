# @title ## 🐘 HQQ

# @markdown See the official [HQQ repository](https://github.com/mobiusml/hqq) for more information.

# !git clone https://github.com/mobiusml/hqq.git
# !pip install -e hqq
# !python hqq/kernels/setup_cuda.py install
# !pip install flash-attn --no-build-isolation
# !pip install transformers --upgrade
# !num_threads=8; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0

import torch
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

BITS = 2 # @param {type:"integer"}
GROUP_SIZE = 128 # @param {type:"integer"}

# Quant config
quant_config = BaseQuantizeConfig(
    nbits=BITS,
    group_size=GROUP_SIZE
)

# Quantize model
model = HQQModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=".",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.quantize_model(quant_config=quant_config, device='cuda')

# Save model and tokenizer
save_folder = MODEL_ID + "-HQQ"
model.save_quantized(save_folder)
tokenizer.save_pretrained(save_folder)

# Upload quant
upload_quant(
    base_model_id=MODEL_ID,
    quantized_model_name=f"{MODEL_NAME}-{BITS}bit-HQQ",
    quantization_type="hqq",
    save_folder=save_folder
)
