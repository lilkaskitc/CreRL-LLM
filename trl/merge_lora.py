from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------
# LOAD MODEL
# ------------------------
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-7B",
    torch_dtype="bfloat16",
    device_map="auto",
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, "./output/trl_grpobaseline_lora2g_5437136")

print("Merging LoRA weights...")
model = model.merge_and_unload()

model.save_pretrained("./output/trl_grpobaseline_lora2g_5437136_merged")
tokenizer = AutoTokenizer.from_pretrained("./output/trl_grpobaseline_lora2g_5437136")
tokenizer.save_pretrained("./output/trl_grpobaseline_lora2g_5437136_merged")