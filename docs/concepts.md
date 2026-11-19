# LangGraph, tool calling and MCP — concept notes

## State
State is the shared data structure carried through graph execution. Nodes read the current state and return updates. In this project, BankingState contains the query, demo account identifier, intent, tool result, final response and execution trace.

## Nodes
A node is one unit of work. It may be a Python function, an LLM call, an API call or business logic. Here, nodes classify a query, execute a mock tool or format a response. Nodes should return state updates rather than mutate the input object.

## Edges
A normal edge always moves execution from one named node to another. For example, the tool node leads to the response-formatting node.

## Conditional edges
A conditional edge calls a routing function and chooses the next node based on its return value. Here, supported account intents route to the tool node; other requests route to a general response.

## Reducers
A reducer is associated with a state key and defines how updates to that key are merged. A key without a custom reducer normally uses replacement semantics. The trace field uses operator.add, so new trace entries are appended to the existing list. Reducers are particularly useful when parallel branches contribute updates to the same field. Choose them carefully; concatenating data is not always the right merge strategy.

## Tool calling
With model tool calling, the model emits a structured request containing a tool name and arguments. The application validates the request, checks authorization and executes the function. The result is returned to the model, which can then produce a final response. The model does not directly execute Python or access a database.

This first milestone uses deterministic keyword routing, not LLM tool calling. That distinction is intentional and should be stated clearly in an interview.

## MCP
MCP (Model Context Protocol) standardizes how an AI host/client discovers and invokes tools exposed by an MCP server.
- Host: the AI application that manages the model and MCP connections.
- MCP client: the component in the host that speaks the protocol to a server.
- MCP server: publishes tools and handles protocol requests.

An application can discover MCP tools and make them available to a model. The model may request a tool call; the host/client invokes it over MCP. MCP is not itself the model's reasoning capability.

### MCP client lifecycle in this repository
1. The MCP host launches the server process with stdio transport.
2. The client and server initialize an MCP session.
3. The client calls `list_tools()` to discover tool names, descriptions and input schemas.
4. The client calls `call_tool()` with a selected tool name and arguments.
5. The server executes its Python function and returns content to the client.

The example in `mcp_client/client.py` demonstrates discovery and invocation independently from the model. The integrated workflow in `mcp_client/agent.py` keeps the MCP session open while LangGraph runs: it discovers tools, converts their JSON schemas into model function tools, lets the model select a tool, invokes that tool through MCP, and returns the result to the model. The MCP server fixes the demo account ID internally, so the model cannot select another account.

### Dynamic MCP tool calling
The key difference from hardcoded tool calling is that the model's available function schemas are built from the MCP server's `list_tools()` response. Adding a new MCP tool can make it available to the agent without manually duplicating its schema in the LLM agent. The host must still validate that the requested tool was actually discovered and handle tool errors.

## Interview explanation
"State carries workflow context, nodes perform individual steps, edges define transitions, and conditional edges implement branching. Reducers define how updates to individual state fields are merged. MCP standardizes tool connectivity, while model tool calling is the model's structured request to use a tool. The host application validates and executes that request."


## Conversation memory and checkpointing

The MCP-connected agent uses LangGraph's `MemorySaver` checkpointer and a `messages` state field with an append reducer.

- A **message** is one user, assistant or tool entry in the conversation.
- A **thread ID** identifies one conversation. Reusing the same thread ID loads its prior checkpointed state.
- A **checkpointer** saves graph state between invocations, so a later user turn can include earlier messages.
- The CLI keeps one graph and MCP session alive and invokes the graph repeatedly with the same thread ID.
- The memory is currently in-process only. It disappears when the CLI exits. A production deployment should use a durable checkpointer, apply retention controls, and isolate thread IDs by authenticated user/session.

For example, after the user asks for an account balance, the next request "show my last two transactions" is sent to the model along with the earlier messages. The model can then interpret follow-up questions using the conversation context.

Checkpointing does not mean the model itself permanently remembers the user. The application retrieves the relevant conversation state and includes it in each model request.
