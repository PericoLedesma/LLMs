# from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#
# import torch
# import sys
# import transformers
#
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from langchain_community.llms import LlamaCpp


# ------ Libraries ------ #

from langchain_community.llms import LlamaCpp

model_llama2_7b = "/Users/pedrorodriguezdeledesmajimenez/scripts/llama2/llama.cpp/models/7B/ggml-model-f16.bin"
model_llama2_13b = "/Users/pedrorodriguezdeledesmajimenez/scripts/llama2/llama.cpp/models/13B/ggml-model-f16.bin"
model_llama2_default = "/Users/pedrorodriguezdeledesmajimenez/scripts/llama2/llama.cpp/models/ggml-vocab-llama.gguf"
model_mistral = "/Users/pedrorodriguezdeledesmajimenez/scripts/GenerativeAI/practice1_local_llm/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


class LLMmodel:
    def __init__(self):
        # LLM = Llama(model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin")

        # self.template = """Question: {question}
        #
        # Answer: Let's work this out in a step by step way to be sure we have the right answer."""
        print('='*40, 'Model initializing...', '='*40)
        self.llm = LlamaCpp(
            model_path=model_llama2_7b,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        print('='*80)


    def llm_response(self, prompt):
        print('User question:', prompt)
        print('LLM working on response ...')
        print('='*80)
        # Make sure the model path is correct for your system!

        prompt = """
        Question: tell me a joke
        """

        response = self.llm(prompt)
        print('='*80)
        print('Model answer: \n', response)
        return response


