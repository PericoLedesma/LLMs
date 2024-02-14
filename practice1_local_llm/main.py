'''
Author: Perico Ledesma
Date: 13-02-2024
Description: Local LLM practice with ctransformers
Link: https://github.com/marella/ctransformers
'''



#  ---------- Libraries ---------- #
from ctransformers import AutoModelForCausalLM

# Paths local models
model_llama2_7b = "../../llama2/llama.cpp/models/7B/ggml-model-f16.bin"
model_llama2_13b = "../../llama2/llama.cpp/models/13B/ggml-model-f16.bin"
model_mistral = "../../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_mistral_q5 = "../../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"



# ------------- MODEL  ---------- #
# Load the model from Hugging Face Hub
# llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml")

# # Load the model from local file
llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model_mistral_q5)

print('Model loaded: ', llm)

prompt = "AI is going to"
print('Inference: ', prompt)
print(llm(prompt))



# for text in llm("AI is going to", stream=True):
#     print(text, end="", flush=True)
#
#
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print(pipe("AI is going to", max_new_tokens=256))

print('Done!')