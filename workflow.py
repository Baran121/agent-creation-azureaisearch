from agent_framework import WorkflowBuilder, WorkflowViz
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.devui import serve
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
import os

# Create your chat client and agents
load_dotenv(".env")
load_dotenv(".secrets.env")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

researcher = AzureOpenAIChatClient(
    
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
).create_agent(
    name="Researcher",
    instructions="you are an assitent and I have attached the knowledge to you, dont use general or any other knowledge'",
    #instructions="you are an personal assistent!'",
    temperature=0.9,
)

writer = AzureOpenAIChatClient(
    
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
).create_agent(
    name="Writer",
    instructions="you are an assitent and I have attached the knowledge to you, dont use general or any other knowledge'",
    #instructions="you are an personal assistent!'",
    temperature=0.9,
)

# Build your workflow
workflow = (
    WorkflowBuilder()
    .set_start_executor(researcher)
    .add_edge(researcher, writer)
    .build()
)

# Convert the workflow to an agent
workflow_agent = workflow.as_agent(name="Content Pipeline Agent")
viz = WorkflowViz(workflow)
if __name__ == "__main__":
    # Host the workflow agent in devui
    viz.save_pdf("workflow.pdf")