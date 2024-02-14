'''
Author: Perico Ledesma
Date: 13-02-2024
Description: Local LLM practice with llama-cpp
Link: https://levelup.gitconnected.com/how-to-run-your-first-local-llms-a5f56a50876e
link: https://medium.com/@cpaggen/minimal-python-code-for-local-llm-inference-112782af509a
https://github.com/abetlen/llama-cpp-python
'''


#  ---------- Libraries ---------- #
from langchain_community.llms import LlamaCpp
import sys

model_llama2_7b = "/Users/pedrorodriguezdeledesmajimenez/scripts/llama2/llama.cpp/models/7B/ggml-model-f16.bin"
model_llama2_13b = "/Users/pedrorodriguezdeledesmajimenez/scripts/llama2/llama.cpp/models/13B/ggml-model-f16.bin"
model_mistral = "../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_mistral_q5 = "../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"


def main():
    llm = LlamaCpp(model_path=model_llama2_7b,
                   # max tokens the model can account for when processing a response
                   # make it large enough for the question and answer
                   n_ctx=4096,
                   # number of layers to offload to the GPU
                   # GPU is not strictly required but it does help
                   n_gpu_layers=32,
                   # number of tokens in the prompt that are fed into the model at a time
                   n_batch=1024,
                   # use half precision for key/value cache; set to True per langchain doc
                   # f16_kv=True,
                   temperature=0.75,
                   max_tokens=2000,
                   top_p=1,
                   verbose=True,  # Verbose is required to pass to the callback manager
                   )

    try:
        print('='*40,"\n\tWelcome to the Local LLM chat\n", 'Model:', llm.model_path,'\n','='*40)

        while True:
            question = input(">>>>> Ask me a question: ")
            if question == "stop":
                sys.exit(1)
            output = llm.invoke(
                question,
                max_tokens=4096,
                temperature=0.2,
                # nucleus sampling (mass probability index)
                # controls the cumulative probability of the generated tokens
                # the higher top_p the more diversity in the output
                top_p=0.1
            )


            print(f"Answer\n>{output}")

    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(1)



    # Gui(page).run(dark_mode=True, title="Local LLM chat with Taipy")

if __name__ == "__main__":
    main()


'''
Other ways to call the model

# Simple inference example
output = llm(
  "<s>[INST] {prompt} [/INST]", # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)

# Chat Completion API

llm = Llama(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ]
)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto")
    sequences = pipeline(
        'Who are the key contributors to the field of artificial intelligence?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
'''