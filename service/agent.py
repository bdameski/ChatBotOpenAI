import os
from typing import Dict, List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from service.agent_inputs_and_tools import (
    NewsToolTopic,
    NewsToolTopicFewShot,
    NewsToolOrganization,
    NewsToolGetOrganizationEmployees,
    NewsToolGetOrganizationsByEmployees,
    NewsToolByCountry,
)

# object to access the open AI LLM
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", streaming=True)
# creating list of the tools that we created, to provide it to the chatbot
tools = [
    NewsToolTopicFewShot(),
    NewsToolOrganization(),
    NewsToolGetOrganizationEmployees(),
    NewsToolGetOrganizationsByEmployees(),
    NewsToolByCountry(),
]

#adding the tools to the LLM object
llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

# Creating prompt for the chatbot to understand its role
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that finds information about news. "
            "If tools require follow up questions, "
            "make sure to ask the user for clarification. Make sure to include any "
            "available options that need to be clarified in the follow up questions. "
            "Do only the things the user specifically requested. Return the responses as "
            "provided by the database.",
        ),
        # keeping the chat history from the user
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        # store intermediate responses, calculations, or logic steps that the agent needs to perform before generating a final response
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# creating the format that the openAI uses to understand chatting history
def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for line in chat_history:
        if line["role"] == "user":
            buffer.append(HumanMessage(content=line["content"]))
        if line["role"] == "assistant":
            buffer.append(AIMessage(content=line["content"]))
    return buffer


# Create agent object as chain.
agent = (
    # Define that when this object is called we need to provide input as "input", chat history as "chat_history".
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: (
            _format_chat_history(x["chat_history"]) if x.get("chat_history") else []
        ),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
    }
    # Add the prompt to it
    | prompt
    # Add the LLM with the tools
    | llm_with_tools
    # USe parser so the chatbot will know how to print the output
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
