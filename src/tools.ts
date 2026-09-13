import OpenAI, { APIError, OpenAIError } from "openai";

export function formatError(error: unknown): string {
  if (error instanceof APIError) {
    return `OpenAI API error (${error.status ?? "unknown status"}): ${error.message}`;
  }
  if (error instanceof OpenAIError) {
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

export type ToolContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };

export type ToolResult = {
  content: ToolContent[];
  isError?: boolean;
};

export async function generateText(args: {
  prompt: string;
  model: string;
  instructions?: string | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.responses.create({
      model: args.model,
      input: args.prompt,
      ...(args.instructions ? { instructions: args.instructions } : {}),
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
}

export async function generateImage(args: {
  prompt: string;
  model: string;
  size?: string | undefined;
  n: number;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.images.generate({
      prompt: args.prompt,
      model: args.model,
      n: args.n,
      ...(args.size ? { size: args.size } : {}),
    });

    const images = response.data ?? [];
    if (images.length === 0) {
      return { content: [{ type: "text", text: "No images were returned." }] };
    }

    const mimeType = `image/${response.output_format ?? "png"}`;
    const content: ToolContent[] = images.map((image) =>
      image.b64_json
        ? { type: "image", data: image.b64_json, mimeType }
        : { type: "text", text: image.url ?? "Image returned with no data." },
    );

    return { content };
  } catch (error) {
    return {
      content: [{ type: "text", text: formatError(error) }],
      isError: true,
    };
  }
}

export async function createEmbeddings(args: {
  input: string | string[];
  model: string;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.embeddings.create({
      input: args.input,
      model: args.model,
    });

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
}

export async function listModels(): Promise<ToolResult> {
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
}
