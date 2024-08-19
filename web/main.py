from typing import Dict, List, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from repository.graph_db import *
from service.agent import agent_executor


if __name__ == "__main__":
    check_graph_db_connection()
    get_entity_types()
    get_graph_schema()
    get_embedding_dimension()


# result = agent_executor.invoke({"input": "What are the health benefits mentioned in the news?"})
# print(result)
