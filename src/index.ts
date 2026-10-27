#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import {
  AUDIO_FORMATS,
  countTokens,
  createEmbeddings,
  createModeration,
  editImage,
  generateImage,
  generateSpeech,
  generateText,
  IMAGE_SIZES,
  listModels,
  parseAllowedOpenAIModels,
  transcribeAudio,
  TRANSCRIPTION_FORMATS,
} from "./tools.js";

function createModelSelectionSchema(
  allowedModels: readonly string[],
  description: string,
) {
  if (allowedModels.includes("*")) {
    return z.string().describe(description);
  }

  const [firstAllowedModel, ...remainingAllowedModels] = allowedModels;
  if (!firstAllowedModel) {
    throw new Error("Cannot create a model schema without an allowed model.");
  }

  return z
    .enum([firstAllowedModel, ...remainingAllowedModels])
    .describe(description);
}

async function main() {
  const allowedOpenAIModels = parseAllowedOpenAIModels();
  const server = new McpServer({
    name: "openai-mcp",
    version: "0.1.0",
  });

  if (allowedOpenAIModels.textModels.length > 0) {
    server.registerTool(
      "generate_text",
      {
        description:
          "Send a prompt to an OpenAI model and return its text reply. Pass images to ask a vision-capable model about them",
        inputSchema: z.object({
          prompt: z.string().describe("The input text to send to the model"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.textModels,
            "OpenAI model to use for text generation. Must support image input if images are provided",
          ),
          instructions: z
            .string()
            .optional()
            .describe(
              "Optional high-level instructions to steer the model's behavior",
            ),
          images: z
            .array(z.string())
            .optional()
            .describe(
              "Images to include with the prompt, each as an http(s) URL or a file path on this server's filesystem (PNG, JPEG, WEBP, or GIF)",
            ),
        }),
      },
      (args) => generateText(args),
    );

    server.registerTool(
      "count_tokens",
      {
        description:
          "Count the input tokens a generate_text request would use, without running the model",
        inputSchema: z.object({
          prompt: z.string().describe("The input text to count"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.textModels,
            "OpenAI model whose tokenizer to use. Must support image input if images are provided",
          ),
          instructions: z
            .string()
            .optional()
            .describe("Optional instructions to include in the count"),
          images: z
            .array(z.string())
            .optional()
            .describe(
              "Images to include in the count, each as an http(s) URL or a file path on this server's filesystem (PNG, JPEG, WEBP, or GIF)",
            ),
        }),
      },
      (args) => countTokens(args),
    );
  }

  if (allowedOpenAIModels.imageModels.length > 0) {
    server.registerTool(
      "generate_image",
      {
        description:
          "Generate an image from a text prompt using an OpenAI image model. By default returns the local file path(s) the image was saved to rather than inline image data; set returnAs to 'base64' for inline data instead",
        inputSchema: z.object({
          prompt: z.string().describe("Description of the image to generate"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.imageModels,
            "OpenAI model to use for image generation",
          ),
          size: z
            .enum(IMAGE_SIZES)
            .optional()
            .describe(
              "Image dimensions. Depends on the model: GPT image models accept 1024x1024, 1536x1024, 1024x1536, or auto; dall-e-2 accepts 256x256, 512x512, or 1024x1024; dall-e-3 accepts 1024x1024, 1792x1024, or 1024x1792",
            ),
          n: z
            .number()
            .int()
            .min(1)
            .max(10)
            .default(1)
            .describe(
              "Number of images to generate (1-10; dall-e-3 only supports 1)",
            ),
          returnAs: z
            .enum(["path", "base64"])
            .default("path")
            .describe(
              "How to return generated images. 'path' (default) saves each image to disk and returns its file path, avoiding the large inline payloads that can exceed an MCP client's tool result size limit. 'base64' returns the image data inline instead, which is more likely to hit that limit for anything but small images",
            ),
          outputPath: z
            .string()
            .optional()
            .describe(
              "File path to save the image to when returnAs is 'path'. Ignored when returnAs is 'base64'. If n > 1, a 1-based index is inserted before the file extension for each image. Defaults to a temp file if omitted",
            ),
        }),
      },
      (args) => generateImage(args),
    );

    server.registerTool(
      "edit_image",
      {
        description:
          "Edit or combine existing images using a text prompt and an OpenAI image model. By default returns the local file path(s) the result was saved to rather than inline image data; set returnAs to 'base64' for inline data instead",
        inputSchema: z.object({
          prompt: z.string().describe("Description of the edit to make"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.imageModels,
            "OpenAI model to use for image editing. GPT image models and dall-e-2 support editing; dall-e-3 does not",
          ),
          images: z
            .array(z.string())
            .min(1)
            .max(16)
            .describe(
              "Images to edit, each as an http(s) URL or a file path on this server's filesystem. GPT image models accept up to 16 PNG, WEBP, or JPEG files under 50MB; dall-e-2 accepts a single square PNG under 4MB",
            ),
          mask: z
            .string()
            .optional()
            .describe(
              "Optional PNG mask, as an http(s) URL or a file path, with the same dimensions as the first image. Fully transparent areas mark where the image should be edited",
            ),
          size: z
            .enum(IMAGE_SIZES)
            .optional()
            .describe(
              "Image dimensions. GPT image models accept 1024x1024, 1536x1024, 1024x1536, or auto; dall-e-2 accepts 256x256, 512x512, or 1024x1024",
            ),
          n: z
            .number()
            .int()
            .min(1)
            .max(10)
            .default(1)
            .describe("Number of edited images to generate (1-10)"),
          returnAs: z
            .enum(["path", "base64"])
            .default("path")
            .describe(
              "How to return edited images. 'path' (default) saves each image to disk and returns its file path, avoiding the large inline payloads that can exceed an MCP client's tool result size limit. 'base64' returns the image data inline instead, which is more likely to hit that limit for anything but small images",
            ),
          outputPath: z
            .string()
            .optional()
            .describe(
              "File path to save the image to when returnAs is 'path'. Ignored when returnAs is 'base64'. If n > 1, a 1-based index is inserted before the file extension for each image. Defaults to a temp file if omitted",
            ),
        }),
      },
      (args) => editImage(args),
    );
  }

  if (allowedOpenAIModels.embeddingModels.length > 0) {
    server.registerTool(
      "create_embeddings",
      {
        description: "Get vector embeddings for one or more pieces of text",
        inputSchema: z.object({
          input: z
            .union([z.string(), z.array(z.string())])
            .describe("Text (or array of texts) to embed"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.embeddingModels,
            "OpenAI model to use for embeddings",
          ),
        }),
      },
      (args) => createEmbeddings(args),
    );
  }

  server.registerTool(
    "list_models",
    {
      description:
        "List available OpenAI models. Returns configured models unless a tool allows any model",
      inputSchema: z.object({}),
    },
    () => listModels(allowedOpenAIModels),
  );

  if (allowedOpenAIModels.speechModels.length > 0) {
    server.registerTool(
      "generate_speech",
      {
        description:
          "Convert text to spoken audio using an OpenAI text-to-speech model. By default returns the local file path the audio was saved to rather than inline data; set returnAs to 'base64' for inline audio instead",
        inputSchema: z.object({
          input: z
            .string()
            .max(4096)
            .describe("The text to convert to speech (max 4096 characters)"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.speechModels,
            "OpenAI model to use for speech generation",
          ),
          voice: z
            .string()
            .default("alloy")
            .describe(
              "Voice to use, e.g. alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse, marin, or cedar",
            ),
          instructions: z
            .string()
            .optional()
            .describe(
              "Optional guidance on tone, style, or delivery. Ignored by tts-1 and tts-1-hd",
            ),
          format: z
            .enum(AUDIO_FORMATS)
            .default("mp3")
            .describe("Audio file format to generate"),
          speed: z
            .number()
            .min(0.25)
            .max(4)
            .optional()
            .describe("Playback speed from 0.25 to 4.0 (default 1.0)"),
          returnAs: z
            .enum(["path", "base64"])
            .default("path")
            .describe(
              "How to return the generated audio. 'path' (default) saves it to disk and returns its file path, avoiding the large inline payloads that can exceed an MCP client's tool result size limit. 'base64' returns the audio data inline instead",
            ),
          outputPath: z
            .string()
            .optional()
            .describe(
              "File path to save the audio to when returnAs is 'path'. Ignored when returnAs is 'base64'. Defaults to a temp file if omitted",
            ),
        }),
      },
      (args) => generateSpeech(args),
    );
  }

  if (allowedOpenAIModels.transcriptionModels.length > 0) {
    server.registerTool(
      "transcribe_audio",
      {
        description:
          "Transcribe spoken audio to text using an OpenAI speech-to-text model",
        inputSchema: z.object({
          file: z
            .string()
            .describe(
              "Audio to transcribe, as an http(s) URL or a file path on this server's filesystem (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm)",
            ),
          model: createModelSelectionSchema(
            allowedOpenAIModels.transcriptionModels,
            "OpenAI model to use for audio transcription",
          ),
          language: z
            .string()
            .optional()
            .describe(
              "ISO-639-1 language code of the input audio (e.g. 'en'). Improves accuracy and latency",
            ),
          prompt: z
            .string()
            .optional()
            .describe(
              "Optional text to guide the model's style or vocabulary, or to continue a previous audio segment",
            ),
          format: z
            .enum(TRANSCRIPTION_FORMATS)
            .default("json")
            .describe(
              "Output format. 'verbose_json' includes segment and word timestamps",
            ),
        }),
      },
      (args) => transcribeAudio(args),
    );
  }

  if (allowedOpenAIModels.moderationModels.length > 0) {
    server.registerTool(
      "create_moderation",
      {
        description:
          "Check text for potentially harmful content using an OpenAI moderation model. Returns, for each input, whether it was flagged plus per-category flags and scores",
        inputSchema: z.object({
          input: z
            .union([z.string(), z.array(z.string())])
            .describe("Text (or array of texts) to classify"),
          model: createModelSelectionSchema(
            allowedOpenAIModels.moderationModels,
            "OpenAI model to use for moderation",
          ),
        }),
      },
      (args) => createModeration(args),
    );
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("openai-mcp server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
