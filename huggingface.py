import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

'''
Hugging face provide two wrappers for LLM. 
1. Models hosted on HuggingFace Hub via API
2. Local pipelines (download models locally). 
'''

customer_email = """
I hope this email finds you amidst an aura of understanding, despite the tangled mess of emotions swirling within me as I write to you. I am writing to pour my heart out about the recent unfortunate experience I had with one of your coffee machines that arrived ominously broken, evoking a profound sense of disbelief and despair.

To set the scene, let me paint you a picture of the moment I anxiously unwrapped the box containing my highly anticipated coffee machine. The blatant excitement coursing through my veins could rival the vigorous flow of coffee through its finest espresso artistry. However, what I discovered within broke not only my spirit but also any semblance of confidence I had placed in your esteemed brand.

Imagine, if you can, the utter shock and disbelief that took hold of me as I laid eyes on a disheveled and mangled coffee machine. Its once elegant exterior was marred by the scars of travel, resembling a war-torn soldier who had fought valiantly on the fields of some espresso battlefield. This heartbreaking display of negligence shattered my dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable
"""  # crea

template = """Given this text, decide what is the issue the customer is concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset

Text: {email}
Category:
"""

prompt = PromptTemplate(template=template, input_variables=["email"])


llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={"temperature":1, "max_length":180}
)
def query(llm, text) -> str:
    return llm(f"Summarize this: {text}!")

query(llm, customer_email)