import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, parse } from "node:path";
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

// Valid sizes depend on the model; see
// https://developers.openai.com/api/reference/resources/images/methods/generate
export const IMAGE_SIZES = [
  "auto",
  "1024x1024",
  "1536x1024",
  "1024x1536",
  "256x256",
  "512x512",
  "1792x1024",
  "1024x1792",
] as const;

export type ToolResult = {
  content: ToolContent[];
  isError?: boolean;
};

// Formats the Responses API accepts for input_image; see
// https://developers.openai.com/api/docs/guides/images-vision
const IMAGE_INPUT_MIME_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

// Local paths are read and inlined as base64 data URLs; http(s) URLs are
// passed straight through so the model fetches them itself.
async function resolveImageUrl(image: string): Promise<string> {
  if (/^https?:\/\//i.test(image)) {
    return image;
  }
  const mimeType = IMAGE_INPUT_MIME_TYPES[extname(image).toLowerCase()];
  if (!mimeType) {
    throw new Error(
      `Unsupported image file extension for "${image}". Supported: ${Object.keys(IMAGE_INPUT_MIME_TYPES).join(", ")}`,
    );
  }
  const data = await readFile(image);
  return `data:${mimeType};base64,${data.toString("base64")}`;
}

export async function generateText(args: {
  prompt: string;
  model: string;
  instructions?: string | undefined;
  images?: string[] | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const input = args.images?.length
      ? [
          {
            role: "user" as const,
            content: [
              { type: "input_text" as const, text: args.prompt },
              ...(await Promise.all(
                args.images.map(async (image) => ({
                  type: "input_image" as const,
                  image_url: await resolveImageUrl(image),
                  detail: "auto" as const,
                })),
              )),
            ],
          },
        ]
      : args.prompt;

    const response = await client.responses.create({
      model: args.model,
      input,
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

// When saving multiple images under one requested outputPath, insert a
// 1-based index before the extension so each image gets a distinct file.
function resolveImagePath(
  outputPath: string | undefined,
  extension: string,
  index: number,
  total: number,
): string {
  if (!outputPath) {
    return join(tmpdir(), `openai-image-${randomUUID()}.${extension}`);
  }
  if (total === 1) {
    return outputPath;
  }
  const { dir, name, ext } = parse(outputPath);
  return join(dir, `${name}-${index + 1}${ext || `.${extension}`}`);
}

export async function generateImage(args: {
  prompt: string;
  model: string;
  size?: string | undefined;
  n: number;
  returnAs: "path" | "base64";
  outputPath?: string | undefined;
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

    const extension = response.output_format ?? "png";
    const mimeType = `image/${extension}`;
    const content: ToolContent[] = await Promise.all(
      images.map(async (image, index): Promise<ToolContent> => {
        if (!image.b64_json) {
          return {
            type: "text",
            text: image.url ?? "Image returned with no data.",
          };
        }
        if (args.returnAs === "base64") {
          return { type: "image", data: image.b64_json, mimeType };
        }
        const filePath = resolveImagePath(
          args.outputPath,
          extension,
          index,
          images.length,
        );
        await mkdir(dirname(filePath), { recursive: true });
        await writeFile(filePath, Buffer.from(image.b64_json, "base64"));
        return { type: "text", text: filePath };
      }),
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
