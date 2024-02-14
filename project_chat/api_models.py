


# class apiLLMmodel_huggingface(object):
#     def __init__(self, filename):
#         self.llm = LlamaCpp(
#             model_path=model_llama2_7b,
#             temperature=0.75,
#             max_tokens=2000,
#             top_p=1,
#             verbose=True,  # Verbose is required to pass to the callback manager
#         )
#         with open(filename, 'r') as file:
#             text = file.read()
#         print('--- Prompt')
#         print(text)
