from dotenv import load_dotenv

load_dotenv()
from typing import Dict, List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings


# Object to access the graph database
graph = Neo4jGraph()
# Object to access the openAI models API to create embeddings
embeddings = OpenAIEmbeddings()

# the index already exists - no expenses for creating embeddings for the existing entities
# The vector index is made on the text property of Chunk entities (property text)
vector_index = Neo4jVector.from_existing_index(embeddings, index_name="news")


def check_graph_db_connection():
    # Cypher query to get one entity from database just to check the connection
    query = """MATCH (organization:Organization) 
    RETURN organization
    LIMIT 1
    """
    # try to execute the code and handle exception if raised
    try:
        # use the object to graph db to execute the query
        response = graph.query(query)
        print(response)
        # get the first organization that is returned
        response = response[0]
        # print the result on the console
        print(response)
    except Exception as e: # if exception is raised, handle it and print it
        raise Exception(f"Error in connecting to the graph database. {str(e)}")

    print("Successfully connected to the graph db.....\n")
    return response


def get_entity_types():
    """Get entity types from graph database to get to know the dataset."""

    # Cypher query to get entity labels
    query = """CALL db.labels() YIELD label
    RETURN collect(label) AS entity_types
    """
    # use the object to graph db to execute the query
    response = graph.query(query)
    # print the response as is returned by Neo4j
    print("Before:", response)
    # Get the first object in the list of results
    response = response[0]
    # the object is of type dict {"key":"value"} -> you have to access it via the key
    response = response["entity_types"]

    print(f"Entity types in graph database: {response}\n")

    return response


def get_graph_schema():
    """Get graph database schema."""
    # use the object to graph db to access the schema of the database
    schema = graph.get_schema
    print(f"Graph schema: {schema}\n")
    return schema


def get_embedding_dimension():
    """Get embedding dimension for embeddings created from OpenAI API."""
    # embed the given query to produce embedding with the use of openAI object that we have created upper
    embedding = embeddings.embed_query("Calculate embedding for this query.")
    # get the lenght of the embedding - list with float numbers 
    dimension = len(embedding)
    print(f"Embedding dimension for openAI embeddings: {dimension}\n")
    return dimension
