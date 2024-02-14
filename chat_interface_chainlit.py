'''
This is a simple chat interface with OpenAI API.

Intro message in chainlit.md

To run thought terminal:
    chainlit run chat_interface_chainlit.py

'''

# Libraries
import os
import chainlit as cl
import openai
from openai import OpenAI

# from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ------------ Setup ------------
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ.get('OPENAI_API_KEY')
)


# question = "What is 1234?"
# template = f"""
# Question: {question}
#  Answer: Let's think step by step.
# """
# template.format(question='what is 1234'))


# ------------ Run when the chat is started ------------
@cl.on_chat_start
def main():
    print("**Chat started**")
    question = "What is 1234?"
    template = f" Question: {question} \n Answer: Let's think step by step."

    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # print("Template started")
    # ll_chain = LLMChain(
    #     prompt=prompt,
    #     llm=client,
    #     verbose=True
    # )
    # print('LLMChain created')
    #
    cl.user_session.set("template", template)


# Run when the user msg is received
@cl.on_message
async def on_message(msg: cl.Message):
    print("+++ The user sent: ", msg.content)
    # template = cl.user_session.get("template")
    # ll_chain = cl.user_session.get("llm_chain")
    # print("ll_chain", ll_chain)
    # question = str(msg.content)
    # print("Complete msg:", template, "---")

    # Response of the model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "content": "You like legos"
            },
            {
                "role": "user",
                "content": msg.content
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Sending back the answer
    print('--- Model answer: ', response.choices[0].message.content)
    await cl.Message(content=f"{response.choices[0].message.content}").send()


