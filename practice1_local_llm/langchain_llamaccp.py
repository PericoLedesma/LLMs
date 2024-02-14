'''
Author: Perico Ledesma
Date: 13-02-2024
Description: Local LLM practice with llama-cpp and langchain
Link: https://python.langchain.com/docs/integrations/llms/llamacpp
'''


#  ---------- Libraries ---------- #
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Paths local models
model_llama2_7b = "../../llama2/llama.cpp/models/7B/ggml-model-f16.bin"
model_llama2_13b = "../../llama2/llama.cpp/models/13B/ggml-model-f16.bin"
model_mistral = "../../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_mistral_q5 = "../../llama2/llama.cpp/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"




template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_mistral,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# streaming = True,
# stream_prefix = True


llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.run(question)


# async def generateStreamingOutput(llm, question):
#     for item in llm.stream(json.dumps(question), stop=['Question:']):
#         yield item
#     return EventSourceResponse(generateStreamingOutput(llm, question))