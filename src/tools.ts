import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, parse } from "node:path";
import OpenAI, { APIError, OpenAIError, toFile, type Uploadable } from "openai";
import type { ImagesResponse } from "openai/resources/images";

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
  | { type: "image"; data: string; mimeType: string }
  | { type: "audio"; data: string; mimeType: string };

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

export type AllowedOpenAIModels = {
  textModels: readonly string[];
  imageModels: readonly string[];
  embeddingModels: readonly string[];
  speechModels: readonly string[];
  transcriptionModels: readonly string[];
  translationModels: readonly string[];
  moderationModels: readonly string[];
};

function parseModelAllowlist(
  modelSetting: string | undefined,
): readonly string[] {
  if (!modelSetting) {
    return [];
  }

  return [
    ...new Set(
      modelSetting
        .split(",")
        .map((modelId) => modelId.trim())
        .filter(Boolean),
    ),
  ];
}

export function parseAllowedOpenAIModels(
  environment: NodeJS.ProcessEnv = process.env,
): AllowedOpenAIModels {
  const {
    OPENAI_MCP_TEXT_MODELS: textModelSetting,
    OPENAI_MCP_IMAGE_MODELS: imageModelSetting,
    OPENAI_MCP_EMBEDDING_MODELS: embeddingModelSetting,
    OPENAI_MCP_SPEECH_MODELS: speechModelSetting,
    OPENAI_MCP_TRANSCRIPTION_MODELS: transcriptionModelSetting,
    OPENAI_MCP_TRANSLATION_MODELS: translationModelSetting,
    OPENAI_MCP_MODERATION_MODELS: moderationModelSetting,
  } = environment;

  return {
    textModels: parseModelAllowlist(textModelSetting),
    imageModels: parseModelAllowlist(imageModelSetting),
    embeddingModels: parseModelAllowlist(embeddingModelSetting),
    speechModels: parseModelAllowlist(speechModelSetting),
    transcriptionModels: parseModelAllowlist(transcriptionModelSetting),
    translationModels: parseModelAllowlist(translationModelSetting),
    moderationModels: parseModelAllowlist(moderationModelSetting),
  };
}

// Formats accepted by the TTS API's response_format; see
// https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
export const AUDIO_FORMATS = [
  "mp3",
  "opus",
  "aac",
  "flac",
  "wav",
  "pcm",
] as const;
export type AudioFormat = (typeof AUDIO_FORMATS)[number];

const AUDIO_MIME_TYPES: Record<AudioFormat, string> = {
  mp3: "audio/mpeg",
  opus: "audio/opus",
  aac: "audio/aac",
  flac: "audio/flac",
  wav: "audio/wav",
  pcm: "audio/pcm",
};

// Formats accepted by the transcription API's response_format; see
// https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
export const TRANSCRIPTION_FORMATS = [
  "json",
  "text",
  "srt",
  "verbose_json",
  "vtt",
] as const;
export type TranscriptionFormat = (typeof TRANSCRIPTION_FORMATS)[number];

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

async function buildResponseInput(
  prompt: string,
  images?: string[] | undefined,
) {
  if (!images?.length) {
    return prompt;
  }
  return [
    {
      role: "user" as const,
      content: [
        { type: "input_text" as const, text: prompt },
        ...(await Promise.all(
          images.map(async (image) => ({
            type: "input_image" as const,
            image_url: await resolveImageUrl(image),
            detail: "auto" as const,
          })),
        )),
      ],
    },
  ];
}

export async function generateText(args: {
  prompt: string;
  model: string;
  instructions?: string | undefined;
  images?: string[] | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.responses.create({
      model: args.model,
      input: await buildResponseInput(args.prompt, args.images),
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

export async function countTokens(args: {
  prompt: string;
  model: string;
  instructions?: string | undefined;
  images?: string[] | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.responses.inputTokens.count({
      model: args.model,
      input: await buildResponseInput(args.prompt, args.images),
      ...(args.instructions ? { instructions: args.instructions } : {}),
    });

    return {
      content: [{ type: "text", text: String(response.input_tokens) }],
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

async function imagesToToolResult(
  response: ImagesResponse,
  returnAs: "path" | "base64",
  outputPath: string | undefined,
): Promise<ToolResult> {
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
      if (returnAs === "base64") {
        return { type: "image", data: image.b64_json, mimeType };
      }
      const filePath = resolveImagePath(
        outputPath,
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

    return await imagesToToolResult(response, args.returnAs, args.outputPath);
  } catch (error) {
    return {
      content: [{ type: "text", text: formatError(error) }],
      isError: true,
    };
  }
}

export async function editImage(args: {
  prompt: string;
  model: string;
  images: string[];
  mask?: string | undefined;
  size?: string | undefined;
  n: number;
  returnAs: "path" | "base64";
  outputPath?: string | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.images.edit({
      prompt: args.prompt,
      model: args.model,
      n: args.n,
      image: await Promise.all(
        args.images.map((image) => resolveUploadableFile(image, "image")),
      ),
      ...(args.mask
        ? { mask: await resolveUploadableFile(args.mask, "image") }
        : {}),
      ...(args.size ? { size: args.size } : {}),
    });

    return await imagesToToolResult(response, args.returnAs, args.outputPath);
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

export async function createModeration(args: {
  input: string | string[];
  model: string;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.moderations.create({
      input: args.input,
      model: args.model,
    });

    const summary = response.results.map(
      ({ flagged, categories, category_scores }) => ({
        flagged,
        categories,
        category_scores,
      }),
    );

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

export async function listModels(
  allowedOpenAIModels: AllowedOpenAIModels,
): Promise<ToolResult> {
  const configuredModelIds = [
    ...new Set(Object.values(allowedOpenAIModels).flat()),
  ].sort();
  const allowsAnyModel = configuredModelIds.includes("*");

  if (allowsAnyModel) {
    try {
      const openAIClient = new OpenAI();
      const availableModelIds: string[] = [];
      for await (const availableModel of openAIClient.models.list()) {
        availableModelIds.push(availableModel.id);
      }
      availableModelIds.sort();
      return {
        content: [
          {
            type: "text",
            text: availableModelIds.join("\n") || "No models available.",
          },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatError(error) }],
        isError: true,
      };
    }
  }

  return {
    content: [
      {
        type: "text",
        text: configuredModelIds.join("\n") || "No models available.",
      },
    ],
  };
}

export async function generateSpeech(args: {
  input: string;
  model: string;
  voice: string;
  instructions?: string | undefined;
  format: AudioFormat;
  speed?: number | undefined;
  returnAs: "path" | "base64";
  outputPath?: string | undefined;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.audio.speech.create({
      input: args.input,
      model: args.model,
      voice: args.voice,
      response_format: args.format,
      ...(args.instructions ? { instructions: args.instructions } : {}),
      ...(args.speed !== undefined ? { speed: args.speed } : {}),
    });

    const buffer = Buffer.from(await response.arrayBuffer());
    if (args.returnAs === "base64") {
      return {
        content: [
          {
            type: "audio",
            data: buffer.toString("base64"),
            mimeType: AUDIO_MIME_TYPES[args.format],
          },
        ],
      };
    }

    const filePath =
      args.outputPath ??
      join(tmpdir(), `openai-speech-${randomUUID()}.${args.format}`);
    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, buffer);
    return { content: [{ type: "text", text: filePath }] };
  } catch (error) {
    return {
      content: [{ type: "text", text: formatError(error) }],
      isError: true,
    };
  }
}

// http(s) URLs are downloaded first since the audio and image upload endpoints
// require an uploaded file rather than a URL reference; local paths are read
// directly.
async function resolveUploadableFile(
  file: string,
  kind: "audio" | "image",
): Promise<Uploadable> {
  if (/^https?:\/\//i.test(file)) {
    const response = await fetch(file);
    if (!response.ok) {
      throw new Error(
        `Failed to download ${kind} from "${file}": ${response.status} ${response.statusText}`,
      );
    }
    const data = Buffer.from(await response.arrayBuffer());
    return toFile(data, basename(new URL(file).pathname) || kind);
  }
  return toFile(await readFile(file), basename(file));
}

function transcriptText(
  response: string | { text: string },
  format: TranscriptionFormat,
): string {
  if (typeof response === "string") {
    return response;
  }
  return format === "verbose_json" ? JSON.stringify(response) : response.text;
}

export async function transcribeAudio(args: {
  file: string;
  model: string;
  language?: string | undefined;
  prompt?: string | undefined;
  format: TranscriptionFormat;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.audio.transcriptions.create({
      file: await resolveUploadableFile(args.file, "audio"),
      model: args.model,
      response_format: args.format,
      ...(args.language ? { language: args.language } : {}),
      ...(args.prompt ? { prompt: args.prompt } : {}),
    });

    return {
      content: [{ type: "text", text: transcriptText(response, args.format) }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: formatError(error) }],
      isError: true,
    };
  }
}

export async function translateAudio(args: {
  file: string;
  model: string;
  prompt?: string | undefined;
  format: TranscriptionFormat;
}): Promise<ToolResult> {
  try {
    const client = new OpenAI();
    const response = await client.audio.translations.create({
      file: await resolveUploadableFile(args.file, "audio"),
      model: args.model,
      response_format: args.format,
      ...(args.prompt ? { prompt: args.prompt } : {}),
    });

    return {
      content: [{ type: "text", text: transcriptText(response, args.format) }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: formatError(error) }],
      isError: true,
    };
  }
}
