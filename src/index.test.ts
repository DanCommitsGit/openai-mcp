import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { z } from "zod";

const registeredToolSchemas = vi.hoisted(() => new Map<string, z.ZodType>());
const modelEnvironmentVariables = [
  "OPENAI_MCP_TEXT_MODELS",
  "OPENAI_MCP_IMAGE_MODELS",
  "OPENAI_MCP_EMBEDDING_MODELS",
  "OPENAI_MCP_SPEECH_MODELS",
  "OPENAI_MCP_TRANSCRIPTION_MODELS",
  "OPENAI_MCP_MODERATION_MODELS",
] as const;
type ModelEnvironmentVariable = (typeof modelEnvironmentVariables)[number];
type ModelEnvironmentOverrides = Partial<
  Record<ModelEnvironmentVariable, string>
>;

vi.mock("@modelcontextprotocol/server", () => ({
  McpServer: class {
    registerTool(name: string, config: { inputSchema: z.ZodType }) {
      registeredToolSchemas.set(name, config.inputSchema);
    }
    async connect() {}
  },
}));

vi.mock("@modelcontextprotocol/server/stdio", () => ({
  StdioServerTransport: class {},
}));

async function loadServerWithModelSettings(
  environmentOverrides: ModelEnvironmentOverrides = {},
) {
  registeredToolSchemas.clear();
  vi.resetModules();
  for (const environmentVariable of modelEnvironmentVariables) {
    vi.stubEnv(environmentVariable, "custom-model");
  }
  for (const [environmentVariable, value] of Object.entries(
    environmentOverrides,
  )) {
    vi.stubEnv(environmentVariable, value);
  }
  await import("./index.js");
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("model arguments", () => {
  it.each([
    ["generate_text", { prompt: "hello" }],
    ["count_tokens", { prompt: "hello" }],
    ["generate_image", { prompt: "a cat" }],
    ["create_embeddings", { input: "hello" }],
    ["generate_speech", { input: "hello" }],
    ["transcribe_audio", { file: "clip.mp3" }],
    ["create_moderation", { input: "hello" }],
  ])("requires an explicit model for %s", async (toolName, toolArguments) => {
    await loadServerWithModelSettings();
    const schema = registeredToolSchemas.get(toolName)!;
    const missingModel = schema.safeParse(toolArguments);
    expect(missingModel.success).toBe(false);
    if (!missingModel.success) {
      expect(
        missingModel.error.issues.map((issue) => issue.path),
      ).toContainEqual(["model"]);
    }
    expect(
      schema.safeParse({ ...toolArguments, model: "custom-model" }).success,
    ).toBe(true);
    expect(
      schema.safeParse({ ...toolArguments, model: "other-model" }).success,
    ).toBe(false);
  });
});

describe("disabled tools", () => {
  it.each([
    ["OPENAI_MCP_TEXT_MODELS", "generate_text"],
    ["OPENAI_MCP_TEXT_MODELS", "count_tokens"],
    ["OPENAI_MCP_IMAGE_MODELS", "generate_image"],
    ["OPENAI_MCP_EMBEDDING_MODELS", "create_embeddings"],
    ["OPENAI_MCP_SPEECH_MODELS", "generate_speech"],
    ["OPENAI_MCP_TRANSCRIPTION_MODELS", "transcribe_audio"],
    ["OPENAI_MCP_MODERATION_MODELS", "create_moderation"],
  ] as const)(
    "omits %s when it is not configured",
    async (environmentVariable, toolName) => {
      await loadServerWithModelSettings({ [environmentVariable]: "" });

      expect(registeredToolSchemas.has(toolName)).toBe(false);
      expect(registeredToolSchemas.has("list_models")).toBe(true);
    },
  );
});

describe("wildcard model settings", () => {
  it.each([
    ["OPENAI_MCP_TEXT_MODELS", "generate_text", { prompt: "hello" }],
    ["OPENAI_MCP_IMAGE_MODELS", "generate_image", { prompt: "a cat" }],
    ["OPENAI_MCP_EMBEDDING_MODELS", "create_embeddings", { input: "hello" }],
    ["OPENAI_MCP_SPEECH_MODELS", "generate_speech", { input: "hello" }],
    [
      "OPENAI_MCP_TRANSCRIPTION_MODELS",
      "transcribe_audio",
      { file: "clip.mp3" },
    ],
    ["OPENAI_MCP_MODERATION_MODELS", "create_moderation", { input: "hello" }],
  ] as const)(
    "allows any model for %s",
    async (environmentVariable, toolName, toolArguments) => {
      await loadServerWithModelSettings({ [environmentVariable]: "*" });

      const schema = registeredToolSchemas.get(toolName)!;
      expect(
        schema.safeParse({ ...toolArguments, model: "any-model" }).success,
      ).toBe(true);
    },
  );
});
