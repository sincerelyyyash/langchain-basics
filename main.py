import os
from dotenv import load_dotenv
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#LLMChain Example usage: 

prompt = PromptTemplate.from_template("What is the capital of {place}?")
chain = LLMChain(llm=llm, prompt= prompt)
# output = chain.run("India")
# print(output)


#Simple Sequential Chain Example usage:
prompt = PromptTemplate.from_template("Give a suitable name for an e commerce store that sells {product}?")
chain1 = LLMChain(llm=llm, prompt= prompt)

prompt = PromptTemplate.from_template("What are the names of specific products at the {store}?")
chain2 = LLMChain(llm=llm, prompt=prompt)

chain = SimpleSequentialChain(
    chains=[chain1, chain2], verbose=True 
)

# output = chain.run("Sneakers")
# print(output)

#Agents and tools example usage:
tools= load_tools(["wikipedia"], llm=llm)
agent = initialize_agent(tools, llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# output= agent.run("When was Shahrukh Khan born?")
# print(output)
