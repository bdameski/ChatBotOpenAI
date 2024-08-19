from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from repository.queries import (
    search_news_by_topic,
    search_by_organization,
    filter_news_by_topic_with_score,
    get_number_employees,
    filter_by_number_employees,
    filter_by_country,
)


fewshot_examples_topic = """{Input:What are the health benefits for employees in the news? Topic: Health benefits}
{Input: Are there any news about new products? Topic: new products}
"""

fewshot_examples = """{Input:What are the health benefits for Google employees in the news? Topic: Health benefits}
{Input: What is the latest positive news about Google? Topic: None}
{Input: Are there any news about VertexAI regarding Google? Topic: VertexAI}
{Input: Are there any news about new products regarding Google? Topic: new products}
"""


# langchain standard for creating Input and Tools, for the chatbot to call them
class NewsInputTopic(BaseModel):
    # define the input, in this case it is topic, and create description which will be used by the chatbot.
    topic: Optional[str] = Field(
        description="Any particular topic that the user wants to finds information for."
    )


class NewsInputTopicFewShot(BaseModel):
    # define the input, in this case it is topic, and create description which will be used by the chatbot.
    # give examples to the chatbot on how to detect the topic from user's question with fewshot examples
    topic: Optional[str] = Field(
        description="Any particular topic that the user wants to finds information for. Here are some examples: "
        + fewshot_examples_topic
    )


class NewsInputOrganization(BaseModel):
    # define the input, in this case it is organization, and create description which will be used by the chatbot.
    organization: Optional[str] = Field(
        description="Organization that the user wants to find information about."
    )


class NewsInputGetOrganizationEmployees(BaseModel):
    # define the input, in this case it is organization, and create description which will be used by the chatbot.
    organization: Optional[str] = Field(
        description="Organization for which the user wants to get number of employees."
    )


class NewsInputFilterOrganizationsByEmployees(BaseModel):
    # define the input, in this case it is number of employees, and create description which will be used by the chatbot.

    number_employees: Optional[int] = Field(
        description="Number of employees by which the user wants to filter organizations."
    )


class NewsInputFilterByCountry(BaseModel):
    # define the input, in this case it is  country name, and create description which will be used by the chatbot.
    country_name: Optional[str] = Field(
        description="The country for which the user wants to find news."
    )

# ------------------ tools:
# this tools are used by the chatbot. It uses them to understand what can be answered by our database, using 
# the queries that we created. Also from here it gets the information what is the input
# that needs to be provided to these functions (queries) to call them
class NewsToolTopic(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationTopic"
    description = "Useful for finding relevant news information on a topic specified by the user."
    args_schema: Type[BaseModel] = NewsInputTopic

    # running in synchronously 
    def _run(
        self, topic: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("Topic extracted:", topic)
        return search_news_by_topic(topic)
    # running in asynchronously -> not using in our usecase
    async def _arun(
        self, topic: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        print("Topic extracted:", topic)
        return search_news_by_topic(topic)


class NewsToolTopicFewShot(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationTopicFewShot"
    description = "Useful for finding relevant news information on a topic specified by the user."
    args_schema: Type[BaseModel] = NewsInputTopicFewShot

    def _run(
        self, topic: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("Topic extracted:", topic)
        return filter_news_by_topic_with_score(topic, 0.85, 0.92)

    async def _arun(
        self, topic: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        print("Topic extracted:", topic)
        return filter_news_by_topic_with_score(topic, 0.85, 0.92)


class NewsToolOrganization(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationOrganization"
    description = (
        "Useful for finding relevant news information about an organization specified by the user."
    )
    args_schema: Type[BaseModel] = NewsInputOrganization

    def _run(
        self,
        organization: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("Organization extracted:", organization)
        return search_by_organization(organization)

    async def _arun(
        self,
        organization: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        print("Organization extracted:", organization)
        return search_by_organization(organization)


class NewsToolGetOrganizationEmployees(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationOrganizationEmployees"
    description = "Useful when the user wants to get number of employees for given organization."
    args_schema: Type[BaseModel] = NewsInputGetOrganizationEmployees

    def _run(
        self,
        organization: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("Organization extracted:", organization)
        return get_number_employees(organization)

    async def _arun(
        self,
        organization: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        print("Organization extracted:", organization)
        return get_number_employees(organization)


class NewsToolGetOrganizationsByEmployees(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationFilterOrganizationsByEmployees"
    description = "Useful when the user wants to filter organizations by the number of employees."
    args_schema: Type[BaseModel] = NewsInputFilterOrganizationsByEmployees

    def _run(
        self,
        number_employees: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("Number extracted:", number_employees)
        return filter_by_number_employees(number_employees)

    async def _arun(
        self,
        number_employees: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        print("Number extracted:", number_employees)
        return filter_by_number_employees(number_employees)


class NewsToolByCountry(BaseTool):
    # the name of the function cannot contain empty spaces
    name = "NewsInformationByCountry"
    description = "Useful when the user wants to filter news by a provided country."
    args_schema: Type[BaseModel] = NewsInputFilterByCountry

    def _run(
        self,
        country_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("country extracted:", country_name)
        return filter_by_country(country_name)

    async def _arun(
        self,
        country_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        print("country extracted:", country_name)
        return filter_by_country(country_name)
