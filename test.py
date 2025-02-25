from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 1. 加载适配器配置
config = PeftConfig.from_pretrained("models")

# 2. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)

# 3. 加载适配器权重到基础模型
model = PeftModel.from_pretrained(base_model, "models")

# 4. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 5. 运行推理
inputs = tokenizer("患者膝盖偶尔锁死，疼痛，然后伴有半月板损伤，其中有一小块软骨组织游离，如何治疗", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=2000)
final = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(final)
