# ------ Libraries ------ #
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Paths local models
model_llama2_7b = "../../llama2/llama.cpp/models/7B/ggml-model-f16.bin"
model_llama2_13b = "../llama2/llama.cpp/models/13B/ggml-model-f16.bin"
model_mistral = "../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_mistral_q5 = "../../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

# Other models: ggml-vocab-gpt2.gguf

class localLLMmodel(object):
    def __init__(self, filename):
        # LLM = Llama(model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin")
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        #
        # n_gpu_layers = 1
        # n_batch = 512


        print('='*40, 'Model initializing...', '='*40)
        self.llm = LlamaCpp(
            model_path=model_llama2_7b,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        print('='*80)
        print(self.llm)
        print('=' * 80)
        with open(filename, 'r') as file:
            self.prompt = file.read()
        print('--- Prompt')
        print(self.prompt)
        print('---------')


    # # Example usage
    # filename = 'example.txt'  # Path to your text file
    # replacements = {'variable': 'replacement_value'}  # Dictionary containing variable-value pairs
    #
    # modified_text = substitute_placeholders(filename, replacements)
    #
    # # Do something with the modified text
    # print(modified_text)



    def llm_response(self, question):
        print('User question:', question)
        print('LLM working on response ...')
        print('='*80)



        prompt = PromptTemplate(template=self.prompt,
                                input_variables=['question'])

        print('Complete prompt:', prompt)

        llm = LLMChain(llm=self.llm, prompt=prompt)
        response = llm.run(question)


        # response = self.llm(prompt)
        print('='*80)
        print('Model answer: \n', response)
        print('=' * 80)
        return response


