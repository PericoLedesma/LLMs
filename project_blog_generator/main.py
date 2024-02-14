import streamlit as st

from llm import *

def main():
    # ----------------- Streamlit App ----------------- #

    st.set_page_config(page_title="Generate Blogs",
                       page_icon='🤖',
                       layout='centered',
                       initial_sidebar_state='collapsed')

    st.title("Chatbot Generator of Blogs 🤖")
    # st.header("Generate Blogs 🤖")

    # Add some text
    st.write("Our innovative Chatbot Generator of Blogs is a cutting-edge tool designed to revolutionize content creation for bloggers and content creators.")
    # Add a divider
    st.markdown("---")

    st.header("<h2> Blog options </h2>")
    # input_text = st.text_input("Enter the Blog Topic")
    input_text = "Artificial intelligence"
    st.write(" > Blog Topic (default): ", input_text)

    selected_option = st.selectbox("\nSelect an option", ["LLama2 7B model", "Llama2 default", "Mistral 7B"])
    st.write(f"You selected: {selected_option}")

    # ---------- History ---------- #
    if "class1" not in st.session_state:
        st.session_state.model = LLMmodel()

    st.markdown("---")
    # Increment the counter when the button is clicked
    if st.button("Click me!"):
        response = st.write(st.session_state.model.llm_response('TODO'))



    # col1, col2 = st.columns([5, 5])
    # with col1:
    #     no_words = st.text_input('No of Words')
    # with col2:
    #     blog_style = st.selectbox('Writing the project_blog_generator for',
    #                               ('Researchers', 'Data Scientist', 'Common People'), index=0)

        # st.write(getLLamaresponse(input_text, no_words, blog_style))


if __name__ == "__main__":
    print('='*40,'\n\tStarting Streamlit App....\n','='*40)
    main()



    # Add a slider
    # age = st.slider("Select your age", 0, 100, 25)
    # st.write(f"Your age is {age} years.")

    # Add a text input
    # name = st.text_input("Enter your name", "John Doe")
    # st.write(f"Hello, {name}!")

    # # Add a checkbox
    # if st.checkbox("Show picture"):
    #     st.image("https://via.placeholder.com/150")

    # # Add a selectbox
    # option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
    # st.write(f"You selected: {option}")

    # # Add a button
    # if st.button("Click me!"):
    #     st.write("Button clicked!")