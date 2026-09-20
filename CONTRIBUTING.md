# Contributing

## Setup

```bash
npm install
npm run build
npm test
```

## Architecture

The codebase is two files by design:

- `src/tools.ts` — tool implementations and allowed-model configuration helpers. OpenAI-backed tools construct their own `OpenAI` client and catch errors, returning `{ content: [...], isError: true }`. `listModels` returns configured IDs locally, or queries OpenAI when a wildcard is configured.
- `src/index.ts` — MCP server wiring. Registers configured tools from `tools.ts` with a Zod `inputSchema`, then connects over stdio. This is the only file that depends on the MCP SDK; `tools.ts` returns a plain result shape and is tested without spinning up a server.

Keep that split when adding or changing tools: OpenAI-calling logic and its return shape stay in `tools.ts`, schema/registration stays in `index.ts`.

## Making changes

- Tests for `tools.ts` live in `src/tools.test.ts` and mock the `openai` package and `node:fs/promises`, so no real API calls or disk writes happen. Tests are appreciated for new behavior, but don't let a missing one hold up a small PR — happy to add coverage during review.
- Run `npm run format` before committing.

## Commit messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, etc.)

## Pull requests

1. Fork the repo and create a branch off `main`.
2. Make sure `npm run format:check`, `npm run build`, and `npm test` all pass.
3. Open a PR describing the change and why it's needed.

CI runs format, build, and test on every PR automatically.
