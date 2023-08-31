from config import *

import json
from typing import Sequence

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import FunctionMessage
from langchain.schema import SystemMessage
from llama_index.tools import BaseTool, FunctionTool

def querytoyindex(query: str) -> str:
    """queries the toys base knowledge and returns the memories from the pinecone index"""
    return query_index.query_index(query,pinecone_namespace_base)


multiply_tool = FunctionTool.from_defaults(fn=querytoyindex)

def queryuserindex(query: str) -> str:
    """queries the users knowledge and returns the memories from the pinecone index"""
    return query_index.query_index(query,pinecone_namespace_user)

add_tool = FunctionTool.from_defaults(fn=queryuserindex)

def talk(query: str) -> str:
    """used when you need to talk to the end user, good to use if no tools are needed, or you need more information from the user"""
    return chat_gpt.ChatGPT(query)

talk_tool = FunctionTool.from_defaults(fn=talk)

class YourOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613"),
        chat_history: ChatMessageHistory = ChatMessageHistory(),
        system_message: SystemMessage = SystemMessage(content="You are a helpful assistant.") 
        
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history
        self._system_message = system_message

    def reset(self) -> None:
        self._chat_history.clear()

    def chat(self, message: str) -> str:
        chat_history = self._chat_history

        # Check if chat history is empty and add system message
        if not chat_history.messages:
            chat_history.add_message(self._system_message)

        chat_history.add_user_message(message)
        functions = [tool.metadata.to_openai_function() for _, tool in self._tools.items()]

        ai_message = self._llm.predict_messages(chat_history.messages, functions=functions)
        chat_history.add_message(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            function_message = self._call_function(function_call)
            chat_history.add_message(function_message)
            ai_message = self._llm.predict_messages(chat_history.messages)
            chat_history.add_message(ai_message)

        return ai_message.content


    def _call_function(self, function_call: dict) -> FunctionMessage:
        tool = self._tools[function_call["name"]]
        output = tool(**json.loads(function_call["arguments"]))
        return FunctionMessage(
            name=function_call["name"],
            content=str(output), 
        )

system_message = SystemMessage(content="You are a helpful assistant.")
agent = YourOpenAIAgent(tools=[multiply_tool, add_tool, talk_tool], system_message=system_message)


exit_conditions = (":q", "quit", "exit")
while True:
    user_input = input("> ")
    if user_input in exit_conditions:
        break
    else:
        print(agent.chat(user_input))

