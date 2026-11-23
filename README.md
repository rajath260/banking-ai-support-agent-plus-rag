# Banking AI Support Agent — Plus RAG

An advanced continuation of [`banking-ai-support-agent`](https://github.com/rajath260/banking-ai-support-agent). This repository starts from the complete, tested LangGraph + MCP + tool-calling + conversation-memory implementation and will extend it with Retrieval-Augmented Generation (RAG).

## Development roadmap
- [x] LangGraph workflows and state management
- [x] Hugging Face model integration (provider/credits required for live inference)
- [x] Validated Python tool calling
- [x] MCP server/client with dynamic tool discovery
- [x] Multi-turn conversation memory with an in-memory checkpointer
- [x] Fictional banking FAQ knowledge base and document loading
- [x] Character-based chunking with overlap and source metadata
- [x] Local text embedding adapter (Sentence Transformers; optional dependency)
- [x] Local Chroma vector storage and semantic retrieval
- [ ] RAG answer generation with source-grounded context
- [ ] Combine RAG retrieval with MCP tools in one agent

> Safety: this is an educational prototype using fictional in-memory banking data. It is not connected to a bank or payment network. Never commit real customer data, credentials, tokens or account details.

---

An educational project for learning Python, LangGraph, Hugging Face, LLM tool calling, MCP and agent architecture through a fictional banking support use case.

> Safety: this project uses fictional in-memory data only. It is not connected to a bank or payment network. Never commit real customer data, credentials, tokens or account details.

## What you will learn
- Difference between an LLM, an agent and a workflow.
- Connecting a Python application to a hosted Hugging Face model.
- Function/tool calling: the model requests a tool; the application validates and executes it.
- LangGraph state, nodes, edges and conditional routing.
- MCP discovery and invocation through a standardized client/server protocol.
- Reliability concepts: validation, retries, checkpointing and human approval.

## Architecture

The project currently provides three runnable workflows:

1. **Basic deterministic workflow:** customer query -> keyword intent classification -> conditional route -> read-only mock banking function or general response.
2. **Direct Hugging Face tool-calling agent:** customer query -> Hugging Face model -> approved Python tool call -> model summarizes the tool result.
3. **MCP-connected Hugging Face agent:** discover MCP tools -> provide their schemas to the model -> invoke the selected tool through the MCP client/server session -> model summarizes the result.

The LLM does not execute Python. It returns a structured tool request. The application validates the tool name and arguments, injects the account ID from application context, executes the allow-listed mock function and sends the result back to the model.

An MCP server exposes mock banking functions over stdio. The standalone MCP client demonstrates discovery and invocation. The MCP-connected LLM agent also discovers tool schemas at runtime, passes them to the Hugging Face model, and invokes the selected tool through the live MCP session. The MCP server fixes the account context to the fictional DEMO-001 account; the model cannot supply an account ID.

## Structure
- app/state.py: state schema for the basic graph
- app/rag/document_loader.py: Markdown loading, chunking and source metadata
- app/rag/embeddings.py: local sentence embedding adapter (optional install via requirements-rag.txt)
- app/rag/vector_store.py: persistent Chroma storage and semantic retrieval (optional install via requirements-rag.txt)
- knowledge_base/: fictional banking FAQ source documents
- tests/test_document_loader.py: offline document-loading and chunking tests
- docs/rag-document-loading-and-chunking.md: RAG Part 1 concepts and limitations
- docs/rag-embeddings.md: RAG Part 2 embedding concepts and setup
- docs/rag-vector-store.md: RAG Part 3 vector storage and semantic retrieval
- app/banking_tools.py: fictional read-only banking functions
- app/graph.py: original deterministic LangGraph workflow
- app/main.py: run the original deterministic workflow
- app/llm.py: Hugging Face hosted chat model connection
- app/agent_tools.py: model tool schemas and safe tool dispatcher
- app/agent_graph.py: LangGraph workflow with model tool calling
- app/agent_main.py: interactive entry point for the direct Python-tool LLM agent
- app/mcp_agent_graph.py: LangGraph workflow using dynamically discovered MCP tools and checkpointed conversation history
- mcp_server/server.py: MCP server exposing mock tools over stdio
- mcp_client/client.py: standalone MCP client discovery/invocation demo
- mcp_client/agent.py: interactive LLM agent connected to the MCP server
- tests/test_graph.py: basic graph tests
- tests/test_agent_tools.py: tool allow-list and argument validation tests
- tests/test_agent_graph.py: offline LangGraph tool-calling tests
- tests/test_mcp_client.py: offline MCP client helper tests
- tests/test_mcp_agent_graph.py: offline tests for the MCP-connected LLM agent
- docs/concepts.md: interview-oriented explanations

## Setup
Python 3.11+ recommended.

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Hugging Face
1. Create a Hugging Face access token with Inference Providers permission at https://huggingface.co/settings/tokens.
2. Create a local `.env` file in the repository root.
3. Add:
   ```env
   HF_TOKEN=hf_your_actual_token
   HF_MODEL=openai/gpt-oss-120b
   ```
4. Never commit `.env`; it is listed in `.gitignore`.

The default model is `openai/gpt-oss-120b`. Availability and tool-calling support depend on the selected inference provider and account.

## Run the Hugging Face connection
```powershell
python -m app.llm
```

## Run the LLM-powered banking agent
```powershell
python -m app.agent_main
```
Example prompts:
- What is my account balance?
- Show my last two transactions.
- Is my debit card active?
- What is the difference between a savings account and a current account?

The direct Python-tool agent receives the account ID from application context. In the MCP-connected agent, the MCP server owns the fixed `DEMO-001` account context; the model does not receive an account-ID argument.

## Run the original deterministic graph
```powershell
python -m app.main
```

## Run tests
```powershell
python -m pytest
```

Tests cover the deterministic graph, banking tool dispatcher, offline LLM-agent graph behavior, MCP client helpers, MCP-connected agent graph and multi-turn conversation history. They do not make a live Hugging Face API call or launch an MCP server process.

## Run the MCP-connected conversational agent
```powershell
python -m mcp_client.agent
```
This workflow requires a working Hugging Face Inference Providers account and available credits. It launches the MCP server, discovers its tools, gives their schemas to the model, invokes any requested tool over MCP, and asks the model to produce the final answer. It keeps conversation history for the lifetime of the running CLI session. Type `exit` or `quit` to end the conversation.

Try this multi-turn sequence:
1. `What is my account balance?`
2. `Show my last two transactions.`
3. `Can you explain the first transaction?`

The third question is sent with the earlier turns in the conversation history. The in-memory checkpointer is reset when the process exits; durable cross-restart memory is a future enhancement. The Hugging Face provider is kept modular and can be replaced later.

## Run the standalone MCP client demo
```powershell
python -m mcp_client.client
```
The client launches the MCP server automatically over stdio, initializes the protocol session, lists the tools published by the server, and invokes the account-balance tool. You should not start the server manually in another terminal for this demo.

To run the server independently for inspection by another MCP host, use:
```powershell
python -m mcp_server.server
```
The server uses stdio, so its standard output is reserved for MCP protocol messages.

## Important concepts
| Concept | Meaning | Project example |
|---|---|---|
| LLM | Model that generates text from input | Hugging Face chat model |
| Tool schema | Description of a function the model may request | `BANKING_TOOLS` or MCP-discovered schemas |
| Tool call | Structured request from the model | function name and JSON arguments |
| Tool dispatcher | Application code that validates and runs a tool | `execute_banking_tool` |
| Node | A unit of work in LangGraph | call model / execute tools / finish |
| Conditional edge | Chooses the next graph node | tool call or direct response |
| MCP | Standard protocol for tool/context exchange | mock banking MCP server |

## Banking security notes
A real banking application needs authentication, authorization at every tool boundary, consent and purpose checks, audit logs, encryption, secrets management, data minimization, rate limiting, monitoring, threat modelling and human approval for sensitive actions. This prototype is read-only and fictional.

## Learning milestones
1. Understand Python tools and LangGraph state/nodes/edges/reducers.
2. Connect a hosted Hugging Face model.
3. Let the model request approved banking tools through function calling (implemented in this milestone).
4. Connect an MCP client and discover/invoke tools (implemented).
5. Add conversation history with LangGraph checkpointing (implemented in-memory).
6. Add retries, durable checkpoint storage, robust error handling and tracing.
7. Load fictional banking FAQs and split them into overlapping chunks (implemented).
8. Generate local embeddings and store vectors for semantic retrieval.
9. Add source-grounded RAG answers and combine retrieval with MCP tools.
8. Prepare architecture diagrams and interview explanations.

## Interview summary
"I built a fictional banking support agent using LangGraph and a Hugging Face hosted model. It discovers read-only tools from an MCP server at runtime, exposes their schemas to the model, validates that requested tools were discovered, invokes them through the MCP client and returns their results to the model. The MCP server owns the fixed demo account context, while LangGraph orchestrates model and tool execution."
