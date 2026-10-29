# openai-mcp

[![npm version](https://img.shields.io/npm/v/%40dancommitsgit%2Fopenai-mcp.svg)](https://www.npmjs.com/package/@dancommitsgit/openai-mcp)
[![CI](https://github.com/DanCommitsGit/openai-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/DanCommitsGit/openai-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Use OpenAI models from Claude, Cursor and any other MCP client: text, images, speech, transcription, embeddings and moderation.

## Features

- Each tool stays off until you set its model variable, so you choose which models the client can use. Only `list_models` is always available.
- Generated and edited images and generated speech are saved to files.
- Images and audio can be given as URLs or local file paths.

## Setup

Requires Node.js 20 or higher and an OpenAI API key.

Each tool is enabled by its own model setting. Omit a setting to leave that tool disabled. Separate multiple model IDs with commas, or use `*` to allow any model. Replace `sk-...` with your OpenAI API key.

```json
{
  "mcpServers": {
    "openai-mcp": {
      "command": "npx",
      "args": ["-y", "@dancommitsgit/openai-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_MCP_TEXT_MODELS": "gpt-5.4-nano",
        "OPENAI_MCP_IMAGE_MODELS": "gpt-image-2.5-flare",
        "OPENAI_MCP_EMBEDDING_MODELS": "text-embedding-3-small",
        "OPENAI_MCP_SPEECH_MODELS": "gpt-4o-mini-tts",
        "OPENAI_MCP_TRANSCRIPTION_MODELS": "gpt-4o-mini-transcribe",
        "OPENAI_MCP_TRANSLATION_MODELS": "whisper-1",
        "OPENAI_MCP_MODERATION_MODELS": "omni-moderation-latest"
      }
    }
  }
}
```

A missing or invalid API key is reported as an error on the first tool call, not when the server starts. A request for a model that is not in the tool's list is rejected.

<details>
<summary>Claude Code</summary>

```bash
claude mcp add --scope user --env OPENAI_API_KEY=sk-... --env OPENAI_MCP_TEXT_MODELS=gpt-5.4-nano --env OPENAI_MCP_IMAGE_MODELS=gpt-image-2.5-flare --transport stdio openai-mcp -- npx -y @dancommitsgit/openai-mcp
```

Add one more `--env` for each other model setting in the JSON above. Keep `--transport stdio` after the last `--env`, or the server name is read as another variable.

This also applies to the Code tab in the Claude desktop app. Drop `--scope user` to make the server available only in the current project.

</details>

<details>
<summary>Claude Desktop</summary>

For the regular chat app. If you use the Code tab in the desktop app, use the Claude Code command instead.

Open Settings → Developer → Edit Config, paste the JSON above into `claude_desktop_config.json`, and restart Claude Desktop. The file is at `~/Library/Application Support/Claude/` on macOS and `%APPDATA%\Claude\` on Windows.

</details>

<details>
<summary>Cursor</summary>

Save the JSON above as `.cursor/mcp.json` in your project, or as `~/.cursor/mcp.json` to use it in every project.

</details>

<details>
<summary>VS Code</summary>

Add the server to `.vscode/mcp.json`. VS Code asks for the API key the first time the server starts and stores it securely. Add other `OPENAI_MCP_*_MODELS` entries as in the JSON above.

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "openai-api-key",
      "description": "OpenAI API key",
      "password": true
    }
  ],
  "servers": {
    "openai-mcp": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@dancommitsgit/openai-mcp"],
      "env": {
        "OPENAI_API_KEY": "${input:openai-api-key}",
        "OPENAI_MCP_TEXT_MODELS": "gpt-5.4-nano",
        "OPENAI_MCP_IMAGE_MODELS": "gpt-image-2.5-flare"
      }
    }
  }
}
```

</details>

### Build from source

```bash
git clone https://github.com/DanCommitsGit/openai-mcp.git
cd openai-mcp
npm install
npm run build
```

Then point your client at the compiled entry point, keeping the same `env` as above:

```json
{
  "command": "node",
  "args": ["/PATH/TO/openai-mcp/build/index.js"]
}
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to run the tests and open a pull request.

## Tools

| Tool                | Needs                             | Description                                                                     |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| `generate_text`     | `OPENAI_MCP_TEXT_MODELS`          | Send a prompt, optional instructions, images, and web search to an OpenAI model |
| `count_tokens`      | `OPENAI_MCP_TEXT_MODELS`          | Count the input tokens a `generate_text` request would use                      |
| `generate_image`    | `OPENAI_MCP_IMAGE_MODELS`         | Generate one or more images from a text prompt                                  |
| `edit_image`        | `OPENAI_MCP_IMAGE_MODELS`         | Edit or combine existing images using a text prompt                             |
| `generate_speech`   | `OPENAI_MCP_SPEECH_MODELS`        | Convert text to spoken audio                                                    |
| `transcribe_audio`  | `OPENAI_MCP_TRANSCRIPTION_MODELS` | Transcribe audio from a URL or local file                                       |
| `translate_audio`   | `OPENAI_MCP_TRANSLATION_MODELS`   | Translate audio from a URL or local file into English text                      |
| `create_embeddings` | `OPENAI_MCP_EMBEDDING_MODELS`     | Get vector embeddings for one or more pieces of text                            |
| `create_moderation` | `OPENAI_MCP_MODERATION_MODELS`    | Check text for potentially harmful content                                      |
| `list_models`       | None                              | List available models                                                           |

Image and speech tools save files to your OS temp directory, unless the client passes `outputPath`, and return the file path.

---

This is an unofficial project and is not affiliated with OpenAI.
