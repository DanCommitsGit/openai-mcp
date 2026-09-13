# openai-mcp

An MCP server that exposes OpenAI API capabilities as tools for an LLM to call.

## Requirements

- Node.js 20 or higher
- An OpenAI API key

## Setup

```bash
npm install
npm run build
```

The server needs `OPENAI_API_KEY` set in its environment. Add it to your MCP client's server configuration, for example:

```json
{
  "mcpServers": {
    "openai-mcp": {
      "command": "node",
      "args": ["/PATH/TO/openai-mcp/build/index.js"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Tools

| Tool | Description |
| --- | --- |
| `generate_text` | Send a prompt to an OpenAI model and return its text reply |
| `generate_image` | Generate an image from a text prompt |
| `create_embeddings` | Get vector embeddings for one or more pieces of text |
| `list_models` | List the OpenAI models available to this API key |

## Development

```bash
npm run build   # compile TypeScript to build/
npm start       # run the compiled server
```
