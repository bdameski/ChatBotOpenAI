from typing import Dict, List, Tuple
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from repository.graph_db import graph, embeddings, vector_index


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~0.8) to each word, then combines them using the AND
    operator. Useful for mapping organization from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    # remove lucene special characters from the input, then split the input into words and save it as list of words
    words = [el for el in remove_lucene_chars(input).split() if el]
    # iterate all of the words ([:-1] means except for the last word ) in order to create the query
    # with sign ~ and the AND operator to search in the database with proxinimity search.
    # If the input is "Amsterdam Inc. Coorporation" then it will produce: Amsterdam~2 AND Inc~ AND Coorporation~2
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"  # ~2 proximity search; indicates that the words should be within two chars of each other.

    full_text_query += f" {words[-1]}~2"
    # remove the empty spaces in the beggining and end of the query
    return full_text_query.strip()


def get_candidates(input: str, candidate_query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve a list of candidate entities from database based on the input string.

    This function queries the Neo4j database using a full-text search. It takes the
    input string, generates a full-text query, and executes this query against the
    specified index in the database. The function returns a list of candidates
    matching the query.
    """
    # Use the upper function to create the query with proximity search, and save it in ft_query variable
    ft_query = generate_full_text_query(input)
    # Use the graph object to execute the query and give parameters: limit and index
    candidates = graph.query(
        candidate_query, {"fulltextQuery": ft_query, "index": 'entity', "limit": limit}
    )
    # If there is direct match in the results from database return only that, otherwise return all options
    direct_match = [
        el["candidate"] for el in candidates if el["candidate"].lower() == input.lower()
    ]
    if direct_match:
        return direct_match

    return [el["candidate"] for el in candidates]


def search_news_by_topic(topic: str):
    """Search for news in database by a topic provided by the user (variable)."""
    # Use the similarity_search function in the vector_index object, that is created upper, which creates
    # embedding of the topic and compares it with the embeddings of the chunk texts
    # Returns the most similar K topics by cosine similarity
    results = vector_index.similarity_search(query=topic, k=2)
    # The result is a list of dict objects with keys:page_content and metadata.
    # Saving only the page_contents in list, as these are the most simiar texts that we want to show.
    page_contents = []
    for r in results:
        print("Page content: ", r.page_content)
        print("Metadata: ", r.metadata)
        # adding new element to the list
        page_contents.append(r.page_content)
    # No need of metadata (decreasing the input tokens to GPT)
    return page_contents


def filter_news_by_topic_with_score(
    topic: str, low_bound_cosine: float = 0, upper_bound_cosine: float = 1
):
    """Same as the upper function, just filters the results by the cosine similarity score."""
    query = """MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article) 
    WITH c.text as chunk_text, vector.similarity.cosine(c.embedding,$embedding) AS score 
    WHERE score <= $upper_bound_cosine AND score >= $low_bound_cosine
    RETURN chunk_text, score
    ORDER BY score DESC
    LIMIT $k
    """
    # Create embedding for the topic entered by the user
    topic_embedding = embeddings.embed_query(topic)
    # We provide query to the graph object to be executed, and parameters are the bounds of the cosine similarity
    # in which we want the result to be.
    results = graph.query(
        query,
        params={
            "low_bound_cosine": low_bound_cosine,
            "upper_bound_cosine": upper_bound_cosine,
            "embedding": topic_embedding,
            "k": 2,
        },
    )
    print("RESULTS: ", results)
    # The result is list of dicts with key chunk_text so we are accessing it.
    results = [r['chunk_text'] for r in results]
    return results


def find_organization(organization: str) -> Tuple[bool, list | str]:
    """Find organizarion in the database which is provided by the user.
    Here we are using the fulltextQuery (with proximity search) that we created above
    in order to get similar organizations if we don't have the exact one in the database.
    (because of typo or it is not part of the database).
    """
    suggested_organizations_query = """
    CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
    YIELD node
    WHERE node:Organization // Filter organization nodes
    RETURN distinct node.name AS candidate
    """
    # Call the function that is created above to get the candidates from database
    candidates = get_candidates(organization, suggested_organizations_query)
    print("Candidates for organization:", candidates)

    # if there are no candidates , return that to the chatbot so it will know.
    if len(candidates) == 0:
        return (
            False,  # The organization does not exist
            "There are not any available organizations with this name. "
            "Ask the user to reconsider the organization.",
        )
    # if there is more than 1 candidate, ask the user which organization it meant.
    if len(candidates) > 1:  # Ask for follow up if too many options
        return (
            False,
            "Ask a follow up question which of the available organizations "
            f"did the user mean. Available options: {candidates}",
        )

    # if there is exact one candidate, return it
    return True, candidates


def search_by_organization(organization: str):
    candidates = find_organization(organization)

    # if there are no candidates or more than 1 candidate for organization, return the message to the chatbot so it will know.
    if candidates[0] == False:
        return candidates[1]
    else:
        candidates = candidates[1]

    # if there is exactly one organization, search for chunk_texts in database for that organization
    organization = candidates[0]
    query = """MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article) 
    WHERE EXISTS {(a)-[:MENTIONS]->(:Organization {name: $organization})}
    WITH c.text as chunk_text
    RETURN chunk_text
    LIMIT $k
    """
    results = graph.query(query, params={"organization": organization, "k": 2})
    print("RESULTS :", results)
    return results


def filter_by_number_employees(number_employees: str):
    """Find organizations which have more than $number_employees, provided as input by the user.
    number_employees is property in the Organization objects.
    """
    query = """MATCH (o:Organization)
    WHERE o.nbrEmployees >= $number_employees
    RETURN o.name as organization_name
    LIMIT $k
    """
    # execute the query in database
    results = graph.query(query, params={"number_employees": number_employees, "k": 5})
    print("RESULTS: ", results)
    return results


def get_number_employees(organization: str):
    """Find number_employees for a given organization, provided as input by the user.
    number_employees is property in the Organization objects.
    """
    # get the candidates for organization name
    candidates = find_organization(organization)

    if candidates[0] == False:
        return candidates[1]
    else:
        candidates = candidates[1]

    query = """MATCH (o:Organization{name:$org_name})
    RETURN o.nbrEmployees as number_employees
    """
    # Create embedding for the topic entered by the user
    results = graph.query(query, params={"org_name": candidates[0]})
    print("RESULTS: ", results)
    return results


def filter_by_country(country_name: str):
    """Find news for a given country, provided as input by the user.
    Articles objects are not directly connected to the country nodes, so we need to traverse the graph.
    """
    query = """MATCH (a:Article)-[:MENTIONS]->(o:Organization)-[:IN_CITY]->(city:City)-[:IN_COUNTRY]->(country:Country{name:$country_name})
    WITH {article_summary: a.summary, organization_name: o.name} as result_dict
    LIMIT $k
    RETURN COLLECT(result_dict) as results
    """
    # Create embedding for the topic entered by the user
    results = graph.query(query, params={"country_name": country_name, "k": 5})
    results = results[0]
    print("RESULTS: ", results)
    return f"Here are article summaries for organizations in {country_name}" + str(results)
