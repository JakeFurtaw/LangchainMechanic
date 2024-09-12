import json
from datetime import datetime
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, Tuple
from langchain.schema.runnable.utils import ConfigurableFieldSpec

CHAT_HISTORY_PATH = "chat_history"

# Ensure the chat history directory exists
os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)

# Initialize the language model
llm = OllamaLLM(
    model="mistral-nemo:latest",
    temperature=0.75,
    num_gpu=2,
)

# Create a prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that is an expert auto mechanic. When a user asks a question, make sure you give a clear and concise answer. If you don't know, just say you don't know."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize memory storage
memory_storage: Dict[Tuple[str, str], ChatMessageHistory] = {}

def save_chat_history(user_id: str, conversation_id: str, history: ChatMessageHistory):
    filename = os.path.join(CHAT_HISTORY_PATH, f"chat_history_{user_id}_{conversation_id}.json")
    with open(filename, 'w') as f:
        json.dump([{"type": msg.type, "content": msg.content, "metadata": str(datetime.now())} for msg in history.messages], f)

def load_chat_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
    filename = os.path.join(CHAT_HISTORY_PATH, f"chat_history_{user_id}_{conversation_id}.json")
    history = ChatMessageHistory()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            messages = json.load(f)
            for msg in messages:
                if msg["type"] == "human":
                    history.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    history.add_ai_message(msg["content"])
    return history

# Function to get or create chat history for a session
def get_session_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
    key = (user_id, conversation_id)
    if key not in memory_storage:
        memory_storage[key] = load_chat_history(user_id, conversation_id)
    return memory_storage[key]

chain = prompt | llm

# Create the conversation chain
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

def get_response(input_text: str, current_user_id: str, current_conversation_id: str):
    print("Assistant: ")

    response = conversation.stream(
        {"input": input_text},
        config={"configurable": {"user_id": current_user_id, "conversation_id": current_conversation_id}}
    )
    full_response = ""
    for chunk in response:
        full_response += chunk
        print(chunk, end="", flush=True)
    print("\n")

    # Save the updated chat history after each response
    save_chat_history(current_user_id, current_conversation_id,
                      get_session_history(current_user_id, current_conversation_id))

    return full_response

# Example usage
GLOBAL_USER_ID = "Jake"
GLOBAL_CONVERSATION_ID = "1"

while True:
    user_input = input(f"User {GLOBAL_USER_ID} (Conversation {GLOBAL_CONVERSATION_ID}): ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Assistant: Goodbye! If you have any more questions about auto mechanics, feel free to ask anytime.")
        break
    get_response(user_input, GLOBAL_USER_ID, GLOBAL_CONVERSATION_ID)