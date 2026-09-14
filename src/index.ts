#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import {
  createEmbeddings,
  generateImage,
  generateText,
  listModels,
} from "./tools.js";

const server = new McpServer({
  name: "openai-mcp",
  version: "0.1.0",
});

server.registerTool(
  "generate_text",
  {
    description: "Send a prompt to an OpenAI model and return its text reply",
    inputSchema: z.object({
      prompt: z.string().describe("The input text to send to the model"),
      model: z
        .string()
        .default("gpt-4o-mini")
        .describe("OpenAI model ID to use"),
      instructions: z
        .string()
        .optional()
        .describe(
          "Optional high-level instructions to steer the model's behavior",
        ),
    }),
  },
  (args) => generateText(args),
);

server.registerTool(
  "generate_image",
  {
    description:
      "Generate an image from a text prompt using an OpenAI image model",
    inputSchema: z.object({
      prompt: z.string().describe("Description of the image to generate"),
      model: z
        .string()
        .default("gpt-image-1")
        .describe("OpenAI image model ID to use"),
      size: z
        .string()
        .optional()
        .describe("Image dimensions, e.g. 1024x1024"),
      n: z
        .number()
        .int()
        .min(1)
        .max(10)
        .default(1)
        .describe("Number of images to generate"),
    }),
  },
  (args) => generateImage(args),
);

server.registerTool(
  "create_embeddings",
  {
    description: "Get vector embeddings for one or more pieces of text",
    inputSchema: z.object({
      input: z
        .union([z.string(), z.array(z.string())])
        .describe("Text (or array of texts) to embed"),
      model: z
        .string()
        .default("text-embedding-3-small")
        .describe("OpenAI embedding model ID to use"),
    }),
  },
  (args) => createEmbeddings(args),
);

server.registerTool(
  "list_models",
  {
    description: "List the OpenAI models available to this API key",
    inputSchema: z.object({}),
  },
  () => listModels(),
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("openai-mcp server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
