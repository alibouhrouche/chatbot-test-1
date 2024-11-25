import os
from datetime import datetime
from operator import itemgetter
from typing import List

import litellm
import streamlit as st
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain.callbacks.base import BaseCallbackHandler
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langfuse.callback import CallbackHandler

litellm.drop_params = True


class MyEmbeddings(Embeddings):
    def __init__(self):
        self.ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef(texts)  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        return self.ef([text])[0]  # type: ignore


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token != "<|im_end|>":
            self.text += token
            self.container.markdown(self.text)


def get_docs(docs: List[Document]):
    out = ""
    for doc in docs:
        question = doc.page_content
        answer = doc.metadata["answer"]
        out += f"Q: {question}\nA: {answer}\n"
    return out


msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

@st.cache_resource(show_spinner=False)
def get_langfuse_handler():
    return CallbackHandler()


view_messages = st.expander("View the message contents in session state")

@st.cache_resource(show_spinner=False)
def initialize_chain():
    prompt_template = """You're a friendly, helpful and easy assistant. Reply in short messages.
For any question related to current events remember that today is {time}

Answer the user query solely based on the information provided in the following questions and answers:
{context}"""

    prompt_value = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "{question}"),
        ]
    )

    def get_time(_):
        return datetime.now().strftime("%a %b %d %Y and the time is %I:%M %p")

    vector_store = Chroma(
        collection_name="docs",
        embedding_function=MyEmbeddings(),
        persist_directory="./chroma"
    )

    model = ChatLiteLLM(
        model="cloudflare/" + os.getenv('CLOUDFLARE_MODEL'),
        streaming=True,
        client=None,
    )

    retriever = vector_store.as_retriever()

    retrieval = RunnableParallel(
        {
            "context": itemgetter("input") | retriever | RunnableLambda(get_docs),
            "question": itemgetter("input"),
            "time": RunnableLambda(get_time),
            "conversation": itemgetter("conversation")
        }
    )

    chain = retrieval | prompt_value | model

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="conversation",
    )


langfuse_handler = get_langfuse_handler()

chain_with_history = initialize_chain()

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(role="assistant", content="Comment puis-je vous aider aujourd'hui ?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        config = {"configurable": {"session_id": "any"}, "callbacks": [langfuse_handler,stream_handler]}
        response = chain_with_history.invoke({"input": prompt}, config)
        st.session_state.messages.append(AIMessage(role="assistant", content=response.content))
