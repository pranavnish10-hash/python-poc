from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.llms import Ollama

# --- Persistent objects for memory and model ---
llm = Ollama(model="gemma3:1b")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Respond directly to the user's question without prefixing your answer with 'AI:' or similar."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{query}")
    ]
)

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# --- Graph state ---
@dataclass
class GraphState:
    user_input: str = ""
    llm_output: str = ""
    final_output: str = ""

# --- Node 1: LLM response ---
def node1(state: GraphState):
    output = conversation_chain.invoke({"query": state.user_input})
    answer = output["text"].lstrip()
    if answer.lower().startswith("ai:"):
        answer = answer[3:].lstrip()
    state.llm_output = answer
    return state

# --- Node 2: Post-process output ---
def node2(state: GraphState):
    # For now, just copy the LLM output
    state.final_output = state.llm_output
    return state

# --- Build the graph ---
graph = StateGraph(GraphState)
graph.add_node("node1", node1)
graph.add_node("node2", node2)
graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", END)
graph = graph.compile()

# --- Main loop for multiple prompts ---
if __name__ == "__main__":
    print("Welcome! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        state = GraphState(user_input=user_input)
        final_state_dict = graph.invoke(state)   # returns dict
        final_state = GraphState(**final_state_dict)
        print("Assistant:", final_state.final_output)
