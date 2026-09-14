#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import {
  AUDIO_FORMATS,
  createEmbeddings,
  generateImage,
  generateSpeech,
  generateText,
  IMAGE_SIZES,
  listModels,
  transcribeAudio,
  TRANSCRIPTION_FORMATS,
} from "./tools.js";

const server = new McpServer({
  name: "openai-mcp",
  version: "0.1.0",
});

server.registerTool(
  "generate_text",
  {
    description:
      "Send a prompt to an OpenAI model and return its text reply. Pass images to ask a vision-capable model about them",
    inputSchema: z.object({
      prompt: z.string().describe("The input text to send to the model"),
      model: z
        .string()
        .default("gpt-4o-mini")
        .describe(
          "OpenAI model ID to use. Must be a vision-capable model (e.g. gpt-4o, gpt-4o-mini) if images are provided. Call list_models to see what's available",
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
  "generate_image",
  {
    description:
      "Generate an image from a text prompt using an OpenAI image model. By default returns the local file path(s) the image was saved to rather than inline image data; set returnAs to 'base64' for inline data instead",
    inputSchema: z.object({
      prompt: z.string().describe("Description of the image to generate"),
      model: z
        .string()
        .default("gpt-image-1")
        .describe(
          "OpenAI image model ID to use, e.g. gpt-image-1, gpt-image-1-mini, gpt-image-1.5, gpt-image-2, dall-e-2, or dall-e-3. Call list_models to see what's available",
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
        .describe(
          "OpenAI embedding model ID to use. Call list_models to see what's available",
        ),
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
      model: z
        .string()
        .default("gpt-4o-mini-tts")
        .describe(
          "OpenAI TTS model ID to use, e.g. gpt-4o-mini-tts, tts-1, or tts-1-hd. Call list_models to see what's available",
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
      model: z
        .string()
        .default("gpt-4o-mini-transcribe")
        .describe(
          "OpenAI speech-to-text model ID to use, e.g. gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe, or whisper-1. Call list_models to see what's available",
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

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("openai-mcp server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
