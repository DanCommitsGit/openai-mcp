# openai-mcp

[![npm version](https://img.shields.io/npm/v/%40dancommitsgit%2Fopenai-mcp.svg)](https://www.npmjs.com/package/@dancommitsgit/openai-mcp)
[![CI](https://github.com/DanCommitsGit/openai-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/DanCommitsGit/openai-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An MCP server that exposes OpenAI API capabilities as tools for an LLM to call.

## Requirements

- Node.js 20 or higher
- An OpenAI API key

## Installation

The simplest way to use the server is with `npx`. Add it to your MCP client's server configuration:

```json
{
  "mcpServers": {
    "openai-mcp": {
      "command": "npx",
      "args": ["-y", "@dancommitsgit/openai-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Tools

| Tool                | Description                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| `generate_text`     | Send a prompt, optional instructions, and optional images to an OpenAI model |
| `generate_image`    | Generate one or more images from a text prompt                               |
| `create_embeddings` | Get vector embeddings for one or more pieces of text                         |
| `list_models`       | List the OpenAI models available to this API key                             |
| `generate_speech`   | Convert text to spoken audio                                                 |
| `transcribe_audio`  | Transcribe audio from a URL or local file                                    |

Generated images and speech are saved to temporary files by default, and the tools return their file paths. Use their `returnAs` option to request inline base64 data instead.

## Development

To run the server from a local clone:

```bash
npm install
npm run build
npm start
```

The server communicates over stdio. For local development, add the path to the compiled entry point to your MCP client's configuration:

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

Run the test suite with:

```bash
npm test
```
