# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import Qwen2Model

# # 加载 Qwen 模型
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", attn_implementation = "eager")

# # 输入 prompt
# input_text = "Once upon a time,"
# inputs = tokenizer(input_text, return_tensors="pt")

# # 不使用 KV Cache 的生成
# output = model.generate(
#     **inputs,
#     max_length=50,
#     use_cache=False,  # 禁用 KV Cache
#     do_sample=True,   # 采样生成
#     top_k=50,         # Top-K 采样
#     temperature=1.0   # 生成的温度
# )

# # 解码输出
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(output_text)

from lightning_fabric import Fabric

fabric = Fabric(accelerator="cpu", devices=2)
fabric.launch()


if fabric.global_rank == 0:
    tensor = 5.0
else:
    tensor = 2.0

tensor = fabric.all_reduce(tensor)
if fabric.global_rank == 0:
    print(tensor)