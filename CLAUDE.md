# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MCP (Model Context Protocol) server that exposes the OpenAI API as tools for an LLM client to call: `generate_text`, `generate_image`, `generate_speech`, `transcribe_audio`, `create_embeddings`, `list_models`. Published to npm as `@dancommitsgit/openai-mcp`; the compiled `build/index.js` is the package's bin entry point.

## Commands

```bash
npm run build         # compile src/ (TypeScript) to build/
npm start              # run the compiled server (build/index.js) over stdio
npm test               # run the full test suite (vitest run)
npx vitest run src/tools.test.ts -t "generateImage"   # run a single test/describe block
npm run format          # prettier --write .
npm run format:check    # prettier --check . (also runs in CI)
```

CI (`.github/workflows/ci.yml`) runs `format:check`, `build`, then `test`, in that order, on every push/PR to `main`. A separate `release` job runs `semantic-release` on manual `workflow_dispatch` — versioning and npm publishing are automated from commit messages (semantic-release conventions), so don't hand-edit the `version` field in `package.json`.

## Architecture

The codebase is two files by design:

- `src/tools.ts` — the actual tool implementations (`generateText`, `generateImage`, `generateSpeech`, `transcribeAudio`, `createEmbeddings`, `listModels`) plus `formatError`. Each tool function constructs its own `new OpenAI()` client (reads `OPENAI_API_KEY` from the environment), calls the corresponding OpenAI SDK method, and never throws: every function catches its own errors and returns `{ content: [...], isError: true }` instead, via the shared `formatError` helper. This lets errors surface to the MCP client as a normal tool result rather than a protocol-level failure.
- `src/index.ts` — the MCP server wiring. Registers each tool from `tools.ts` with its Zod `inputSchema` on an `McpServer`, then connects over `StdioServerTransport`. This is the only file that knows about the MCP SDK; `tools.ts` has no MCP dependency and returns a plain `ToolResult` (`{ content: ToolContent[], isError?: boolean }`), which is why it's tested directly without spinning up a server.

Keep this separation: schema/registration concerns stay in `index.ts`, OpenAI-calling logic and its return shape stay in `tools.ts`.

### Image generation specifics

`generate_image` defaults to `returnAs: "path"` (saves the image to disk and returns the file path) rather than inline base64, specifically to avoid exceeding an MCP client's tool-result size limit — keep this default when touching that tool. `resolveImagePath` in `tools.ts` handles the multi-image case by inserting a 1-based index before the file extension when `n > 1` and an explicit `outputPath` is given. Valid `size` values are enum-checked against `IMAGE_SIZES` and vary by model (GPT image models vs. `dall-e-2` vs. `dall-e-3`).

### Audio specifics

`generate_speech` (TTS) follows the same `returnAs: "path"` default as `generate_image`, for the same tool-result-size reason. `transcribe_audio` (STT) takes an http(s) URL or a local file path; a URL is downloaded with `fetch` first since the OpenAI transcription endpoint requires an uploaded file rather than a URL reference, then both cases go through the SDK's `toFile` helper. Valid values for `generate_speech`'s `format` and `transcribe_audio`'s `format` are enum-checked against `AUDIO_FORMATS` and `TRANSCRIPTION_FORMATS` respectively.

## Testing conventions

Tests live alongside source as `*.test.ts` (excluded from the TS build via `tsconfig.json`). `src/tools.test.ts` mocks the `openai` module and `node:fs/promises` with `vi.hoisted`/`vi.mock` so no real API calls or disk writes happen; each tool function is tested directly against those mocks rather than through the MCP server.

## OpenAI API documentation

When implementing or changing OpenAI SDK calls (params, response shapes, model/size enums, new endpoints), verify against current OpenAI docs rather than training-data memory:

- https://developers.openai.com/api/docs/llms.txt — index of guides (concepts, migrations, GPT Actions, changelog, etc.)
- https://developers.openai.com/api/reference/llms.txt — index of the full API reference (endpoints grouped by resource, e.g. audio, batches, chat completions)

Each is an `llms.txt` index of linked markdown pages — fetch the index first, then follow the specific page relevant to the task.

## Commit attribution

Add a `Co-Authored-By: Claude <noreply@anthropic.com>` trailer to a commit when Claude wrote a significant share of the changed files. Omit it when the user has fully reviewed and understands the resulting code, or was significantly involved in planning/designing the change — that marks the commit as human-owned rather than vibe-coded.
