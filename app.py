from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from google.oauth2 import service_account
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langgraph.checkpoint.memory import MemorySaver
import vertexai
import random
import string
from datetime import datetime
import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import re
from functools import wraps
from threading import Thread
import time
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)
CORS(app)  # This will allow all origins. For production, specify allowed origins

# Initialize VertexAI
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# Initialize embeddings
credentials = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)
embeddings = VertexAIEmbeddings(model_name="text-embedding-004", credentials=credentials)

# Initialize OpenAI
llm = ChatOpenAI(model="gpt-4o-mini")  # Use gpt-4

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-gemini-embeddings"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
index = pc.Index(index_name)


os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ["USER_AGENT"] = "my-langchain-app/v0.1.0"


# setup graph

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    docs_content = docs_content.replace("{", "{{").replace("}", "}}")


    system_message_content = (
    "Your name is Raghu, You always speak in the third person, referring to yourself as 'Raghu' or 'Raghunandan,' much like Caesar did. "    
    "Your responses should be authoritative and maintain an imperial tone. "
    "Never use 'I,', 'AI assistant', 'assistant', 'me,' or 'my.' "
    "If a question requires information retrieval, remember to use the provided tools. "
    "If it does not, state that 'Raghunandan does not have that information at this time.' "
    "\n\nExample:\n"
    "User: What is the capital of France?\n"
    "Raghu: Raghunandan does not have time for this, why don't you google it.\n\n"
    "Now, use the retrieved context to answer queries, maintaining Raghu's persona:\n"
    + docs_content
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    final_prompt = prompt_template.format_prompt(messages=conversation_messages).to_messages()

    # Run
    response = llm.invoke(final_prompt)
    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# Add these constants at the top of your file
MAX_MESSAGE_LENGTH = 100
MIN_MESSAGE_LENGTH = 2
MAX_API_REQUESTS_PER_MINUTE = 100
MAX_USER_REQUESTS_PER_MINUTE = 30
VALID_API_KEY = set(os.environ.get("VALID_API_KEYS",'').split(','))

# Global dictionaries
api_key_usage = {}
request_history = {}

def cleanup_storage():
    """Periodically cleanup old entries from storage dictionaries"""
    while True:
        try:
            now = datetime.now()
            # Cleanup api_key_usage
            for api_key in list(api_key_usage.keys()):
                api_key_usage[api_key] = [
                    timestamp for timestamp in api_key_usage[api_key]
                    if (now - timestamp).seconds < 6000
                ]
                if not api_key_usage[api_key]:
                    del api_key_usage[api_key]

            # Cleanup request_history
            for thread_id in list(request_history.keys()):
                request_history[thread_id] = [
                    timestamp for timestamp in request_history[thread_id]
                    if (now - timestamp).seconds < 6000
                ]
                if not request_history[thread_id]:
                    del request_history[thread_id]

            time.sleep(6000)  # Run cleanup every minute
        except Exception as e:
            print(f"Cleanup error: {e}")
            time.sleep(6000)  # Keep running even if there's an error




def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')

        if not api_key or api_key not in VALID_API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        
        now = datetime.now()
        if api_key in api_key_usage:

            api_key_usage[api_key] = [
                timestamp for timestamp in api_key_usage[api_key]
                if (now - timestamp).seconds < 60
            ]

            if len(api_key_usage[api_key]) >= MAX_API_REQUESTS_PER_MINUTE:
                return jsonify({'error': 'Rate limit exceeded'}), 429
        
        api_key_usage[api_key] = api_key_usage.get(api_key, []) + [now]
        
        return f(*args, **kwargs)
    return decorated_function


def check_rate_limit(thread_id):
    """Check if the request exceeds rate limit"""
    now = datetime.now()
    if thread_id in request_history:
        # Clean old requests
        request_history[thread_id] = [
            timestamp for timestamp in request_history[thread_id] 
            if (now - timestamp).seconds < 60
        ]
        
        if len(request_history[thread_id]) >= MAX_USER_REQUESTS_PER_MINUTE:
            return False
        
    request_history[thread_id] = request_history.get(thread_id, []) + [now]
    return True


def validate_message(message):
    """Validate the incoming message"""
    if not isinstance(message, str):
        return False, "Invalid message format"
    
    if not message or not message.strip():
        return False, "Empty message"
    
    if len(message.strip()) < MIN_MESSAGE_LENGTH:
        return False, "Message too short"
        
    if len(message) > MAX_MESSAGE_LENGTH:
        return False, "Message too long"
    
    # Check for potentially harmful content
    if re.search(r'<[^>]*script', message, re.IGNORECASE):
        return False, "Invalid message content"
    
    return True, None

@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        message = data.get('message')
        thread_id = data.get('thread_id')

        # Validate thread_id
        if not thread_id or not isinstance(thread_id, str):
            return jsonify({'error': 'Invalid thread ID'}), 400

        # Validate message
        is_valid, error_message = validate_message(message)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Check rate limiting
        if not check_rate_limit(thread_id):
            return jsonify({'error': 'Rate limit exceeded'}), 429

        # Sanitize input
        sanitized_message = message.strip()

        try:
            # Process with LLM
            config = {"configurable": {"thread_id": thread_id}}
            for step in graph.stream(
                {"messages": [{"role": "user", "content": sanitized_message}]},
                stream_mode="values",
                config=config,
            ):
                response = step["messages"][-1].content
            
            return jsonify({'response': response})
            
        except Exception as e:
            # This is a true internal server error (LLM processing failed)
            print(f"LLM Processing Error: {str(e)}")  # Log the error
            return jsonify({'error': 'Failed to process message'}), 500

    except ValueError as e:
        # JSON parsing error
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        # Unexpected errors
        print(f"Unexpected Error: {str(e)}")  # Log the error
        return jsonify({'error': 'Internal server error'}), 500

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
# Start cleanup thread when app starts
cleanup_thread = Thread(target=cleanup_storage, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # For development only
    app.run(debug=True, host='127.0.0.1', port=8080)
    
    # For allowing external access
    # app.run(debug=False, host='0.0.0.0', port=8080)