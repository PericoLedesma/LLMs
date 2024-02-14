import os
import shutil
from langchain_community import document_loaders as dl
from langchain import text_splitter as ts
from langchain_community import embeddings
from langchain_community import vectorstores as vs
from langchain import retrievers
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


# Some constant
DS_TYPE_LIST = [".WEB", ".PDF", ".TXT"]
SPLIT_TYPE_LIST = ["CHARACTER", "TOKEN"]
EMBEDDING_TYPE_LIST = ["HF", "OPENAI"]
VECTORSTORE_TYPE_LIST = ["FAISS", "CHROMA", "SVM"]
REPO_ID_DEFAULT = "declare-lab/flan-alpaca-large"
CHAIN_TYPE_LIST = ["stuff", "map_reduce", "map_rerank", "refine"]


class DocumentPipeline(object):
    """
    TalkDocument is a class for processing and interacting with documents, embeddings, and question-answering chains.
    Attributes:
        data_dir_path (str): Path to the data source (TXT, PDF, or web URL).
        HF_API_TOKEN (str): Hugging Face API token.
        OPENAI_KEY (str): OpenAI API key.
        document (str): Loaded document content.
        document_splited (list): List of document chunks after splitting.
        embedding_model (EmbeddingsBase): Embedded model instance.
        embedding_type (str): Type of embedding model used (HF or OPENAI).
        db (VectorStoreBase): Vector storage instance.
        llm (HuggingFaceHub): Hugging Face Hub instance.
        chain (QuestionAnsweringChain): Question answering chain instance.
        repo_id (str): Repository ID for Hugging Face models.
    Methods:
        get_document(data_source_type="TXT"): Load the document content based on the data source type.
        get_split(split_type="character", chunk_size=1000, chunk_overlap=10): Split the document content into chunks.
        get_embedding(embedding_type="HF", OPENAI_KEY=None): Get the embedding model based on the type.
        get_storage(vectorstore_type="FAISS", embedding_type="HF", OPENAI_KEY=None): Create vector storage using embeddings.
        get_search(question, with_score=False): Perform a similarity search for relevant documents.
        do_question(question, repo_id="declare-lab/flan-alpaca-large", chain_type="stuff", relevant_docs=None, with_score=False, temperature=0, max_length=300, language="Spanish"): Answer a question using relevant documents and a question-answering chain.
        create_db_document(data_source_type="TXT", split_type="token", chunk_size=200, embedding_type="HF", chunk_overlap=10, OPENAI_KEY=None, vectorstore_type="FAISS"): Create and return a vector storage instance with document content.
    """

    def __init__(self, HF_API_TOKEN, OPENAI_KEY, data_dir_path, db_dir_path, archive_path) -> None:
        """
        Initialize the TalkDocument instance.
        :param data_dir_path: Path to the data source (TXT, PDF, or web URL).
        :type data_dir_path: str
        :param HF_API_TOKEN: Hugging Face API token.
        :type HF_API_TOKEN: str
        :param OPENAI_KEY: OpenAI API key.
        :type OPENAI_KEY: str, optional
        """
        # PATHS
        self.data_dir_path = data_dir_path
        self.db_dir_path = db_dir_path
        self.archive_path = archive_path

        self.document = None
        self.document_splited = None
        self.embedding_model = None
        self.embedding_type = None
        self.OPENAI_KEY = OPENAI_KEY
        self.HF_API_TOKEN = HF_API_TOKEN
        self.db = None
        self.llm = None
        self.chain = None
        self.repo_id = None

        if not self.data_dir_path:
            # TODO ADD LOGS
            print("YOU MUST INTRODUCE ONE OF THEM")
        print('DocumentPipeline instance created')


    def get_document(self):
        """
        Load the document content based on the data source type.
        :param data_source_type: Type of data source (TXT, PDF, WEB).
        :type data_source_type: str, optional
        :return: Loaded document content.
        :rtype: str
        """
        document = None

        print('==== Loading document...')
        for filename in os.listdir(self.data_dir_path):
            if os.path.isfile(os.path.join(self.data_dir_path, filename)):
                print("Found document:", filename)

                _, data_source_type = os.path.splitext(filename)
                data_source_type = data_source_type.upper() if data_source_type.upper() in DS_TYPE_LIST else \
                DS_TYPE_LIST[0]

                if data_source_type == ".TXT":
                    loader = dl.TextLoader(os.path.join(self.data_dir_path, filename))
                    document = loader.load()

                elif data_source_type == ".PDF":
                    loader = dl.PyPDFLoader(os.path.join(self.data_dir_path, filename))
                    document = loader.load()

                elif data_source_type == ".WEB":
                    loader = dl.WebBaseLoader(os.path.join(self.data_dir_path, filename))
                    document = loader.load()
                else:
                    print("Error: Data source type not supported")

                if self.document:
                    self.document.extend(document)
                else:
                    self.document = document


                print(f'File loaded!. Lenght of the document(Pages loaded): {len(self.document)}')
                shutil.move(os.path.join(self.data_dir_path, filename), self.archive_path)


    def get_split(self, split_type, chunk_size, chunk_overlap):
        """
        Split the document content into chunks.
        :param split_type: Type of splitting (character, token).
        :type split_type: str, optional
        :param chunk_size: Size of each chunk.
        :type chunk_size: int, optional
        :param chunk_overlap: Overlap size between chunks.
        :type chunk_overlap: int, optional
        :return: List of document chunks after splitting.
        :rtype: list
        """
        print('==== Splitting text...', end=' ')
        split_type = split_type.upper() if split_type.upper() in SPLIT_TYPE_LIST else SPLIT_TYPE_LIST[0]

        if self.document:
            if split_type == "CHARACTER":
                text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif split_type == "TOKEN":
                text_splitter = ts.TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                print("Error: Split type not supported")

            try:
                self.document_splited = text_splitter.split_documents(documents=self.document)
                print('Splitted into', len(self.document_splited), 'chunks')
            except Exception as error:
                print(f"Error in split data source step: {error}")

            # return self.document_splited
        else:
            print("Error: You need to load the document first")


    def get_embedding(self):
        """
        Get the embedding model based on the type.
        :param embedding_type: Type of embedding model (HF, OPENAI).
        :type embedding_type: str, optional
        :param OPENAI_KEY: OpenAI API key.
        :type OPENAI_KEY: str, optional
        :return: Embedded model instance.
        :rtype: EmbeddingsBase
        """
        if not self.embedding_model:
            if self.embedding_type == "HF":
                self.embedding_model = embeddings.HuggingFaceEmbeddings()

            elif self.embedding_type == "OPENAI":
                if self.OPENAI_KEY:
                    self.embedding_model = embeddings.OpenAIEmbeddings(openai_api_key=self.OPENAI_KEY)
                else:
                    print("You need to introduce a OPENAI API KEY")

            self.embedding_type = self.embedding_type

            return self.embedding_model


    def get_storage(self, vectorstore_type, embedding_type):
        """
        Create vector storage using embeddings.
        :param vectorstore_type: Type of vector storage (FAISS, CHROMA, SVM).
        :type vectorstore_type: str, optional
        :param embedding_type: Type of embedding model (HF, OPENAI).
        :type embedding_type: str, optional
        :param OPENAI_KEY: OpenAI API key.
        :type OPENAI_KEY: str, optional
        :return: Vector storage instance.
        :rtype: VectorStoreBase
        """
        print('==== Creating Vector storage... ')
        if not os.path.exists(self.db_dir_path):
            os.makedirs(self.db_dir_path) # Persist directory

        vectorstore_type = vectorstore_type.upper() if vectorstore_type.upper() in VECTORSTORE_TYPE_LIST else VECTORSTORE_TYPE_LIST[0]
        self.embedding_type = embedding_type.upper() if embedding_type.upper() in EMBEDDING_TYPE_LIST else EMBEDDING_TYPE_LIST[0]

        self.get_embedding()

        if vectorstore_type == "FAISS":
            model_vectorstore = vs.FAISS

        elif vectorstore_type == "CHROMA":
            model_vectorstore = vs.Chroma

        elif vectorstore_type == "SVM":
            model_vectorstore = retrievers.SVMRetriever
        else:
            print("Error: Vector storage type not supported")

        # TODO
        # elif vectorstore_type == "LANCE":
        #     model_vectorstore = vs.LanceDB
        if self.document_splited:
            try:
                self.db = model_vectorstore.from_documents(documents=self.document_splited,
                                                           embedding=self.embedding_model,
                                                           persist_directory=self.db_dir_path)
                self.db.persist()

            except Exception as error:
                print(f"Error in storage data source step: {error}")
                self.db = None
        else:
            # Now we can load the persisted database from disk, and use it as normal.
            self.db = model_vectorstore(persist_directory=self.db_dir_path,
                                         embedding_function=self.embedding_model)

        print('Vector storage created: ', type(self.db))


    def get_search(self, question, with_score=False):
        """
        Perform a similarity search for relevant documents.
        :param question: Question text.
        :type question: str
        :param with_score: Flag indicating whether to include relevance scores.
        :type with_score: bool, optional
        :return: Relevant documents or document indices.
        :rtype: list or ndarray
        """

        # TODO MultiQueryRetriever AND Max marginal relevance
        print(f'Searching relevant chunks in Database {str(type(self.db))}...', end=' ')

        relevant_docs = None

        if self.db and "SVM" not in str(type(self.db)):

            if with_score:
                # Return docs and relevance scores in the range [0, 1]. Defaults k= 4.
                relevant_docs = self.db.similarity_search_with_relevance_scores(question, k=5)
            else:
                relevant_docs = self.db.similarity_search(question)
        elif self.db:
            relevant_docs = self.db.get_relevant_documents(question)
        else:
            print("Error: You need to create the vector storage first")

        print('Done!' if relevant_docs else 'Error: No relevant documents found')
        return relevant_docs

    def do_question(self, question, relevant_docs, with_score):
        """
        Answer a question using relevant documents and a question-answering chain.
        :param question: Question text.
        :type question: str
        :param relevant_docs: Relevant documents or document indices.
        :type relevant_docs: list or ndarray, optional
        :param with_score: Flag indicating whether to include relevance scores.
        :type with_score: bool, optional
        """
        print('=== Searching related documentation...')
        relevant_docs = self.get_search(question, with_score=with_score)

        # self.chain = self.chain if self.chain is not None else load_qa_chain(self.llm, chain_type=chain_type,
        #                                                                      prompt=PROMPT)
        # response = self.chain({"input_documents": relevant_docs, "question": question}, return_only_outputs=True)

        return relevant_docs


    def create_db_document(self,
                           split_type,
                           chunk_size,
                           embedding_type,
                           chunk_overlap,
                           vectorstore_type):
        """
        Create and return a vector storage instance with document content.
        :param split_type: Type of splitting (token, character).
        :type split_type: str, optional
        :param chunk_size: Size of each chunk.
        :type chunk_size: int, optional
        :param embedding_type: Type of embedding model (HF, OPENAI).
        :type embedding_type: str, optional
        :param chunk_overlap: Overlap size between chunks.
        :type chunk_overlap: int, optional
        :param vectorstore_type: Type of vector storage (FAISS, CHROMA, SVM).
        :type vectorstore_type: str, optional
        :return: Vector storage instance.
        :rtype: VectorStoreBase
        """

        self.get_document()
        if self.document:
            self.get_split(split_type=split_type,
                           chunk_size=chunk_size,
                           chunk_overlap=chunk_overlap)
        self.get_storage(vectorstore_type=vectorstore_type,
                          embedding_type=embedding_type)



# ********************************** EXAMPLE **********************************
# obj = TalkDocument(HF_API_TOKEN = "YOURKEY",data_source_path="data/test.txt")
# obj.create_db_document()

# question = "What is Hierarchy 4.0?"
# res  = obj.do_question(question=question, language="ENGLISH")
# print(res)

"""
RESPONSE: 
{'output_text': "Hierarchy 4.0 is an innovative software solution for control Safety Systems. It provides an interactive diagram of the entire plant revealing cause and effect Behavior with readings provided in a hierarchical view allowing for a deep understanding of the system's strategy. All data is collected from multiple sources visualized as a diagram and optimized through a customized dashboard allowing users to run a logic simulation from live data or pick a moment from their history. Your simulation is based on actual safety Logics not just on a math model."}
"""
