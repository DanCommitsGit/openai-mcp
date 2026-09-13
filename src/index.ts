import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import OpenAI, { APIError, OpenAIError } from "openai";
import { z } from "zod";

const server = new McpServer({
  name: "openai-mcp",
  version: "0.1.0",
});

function formatError(error: unknown): string {
  if (error instanceof APIError) {
    return `OpenAI API error (${error.status ?? "unknown status"}): ${error.message}`;
  }
  if (error instanceof OpenAIError) {
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

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
  async ({ prompt, model, instructions }) => {
    try {
      const client = new OpenAI();
      const response = await client.responses.create({
        model,
        input: prompt,
        ...(instructions ? { instructions } : {}),
      });

      return {
        content: [
          { type: "text", text: response.output_text || "No response content." },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatError(error) }],
        isError: true,
      };
    }
  },
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
  async ({ prompt, model, size, n }) => {
    try {
      const client = new OpenAI();
      const response = await client.images.generate({
        prompt,
        model,
        n,
        ...(size ? { size } : {}),
      });

      const images = response.data ?? [];
      if (images.length === 0) {
        return { content: [{ type: "text", text: "No images were returned." }] };
      }

      const mimeType = `image/${response.output_format ?? "png"}`;
      const content = images.map((image) =>
        image.b64_json
          ? ({ type: "image" as const, data: image.b64_json, mimeType })
          : ({
              type: "text" as const,
              text: image.url ?? "Image returned with no data.",
            }),
      );

      return { content };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatError(error) }],
        isError: true,
      };
    }
  },
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
  async ({ input, model }) => {
    try {
      const client = new OpenAI();
      const response = await client.embeddings.create({ input, model });

      const summary = response.data.map((embedding) => ({
        index: embedding.index,
        embedding: embedding.embedding,
      }));

      return {
        content: [{ type: "text", text: JSON.stringify(summary) }],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatError(error) }],
        isError: true,
      };
    }
  },
);

server.registerTool(
  "list_models",
  {
    description: "List the OpenAI models available to this API key",
    inputSchema: z.object({}),
  },
  async () => {
    try {
      const client = new OpenAI();
      const modelIds: string[] = [];
      for await (const model of client.models.list()) {
        modelIds.push(model.id);
      }
      modelIds.sort();

      return {
        content: [
          { type: "text", text: modelIds.join("\n") || "No models available." },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatError(error) }],
        isError: true,
      };
    }
  },
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
