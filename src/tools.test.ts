import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  responsesCreate: vi.fn(),
  imagesGenerate: vi.fn(),
  embeddingsCreate: vi.fn(),
  modelsList: vi.fn(),
  speechCreate: vi.fn(),
  transcriptionsCreate: vi.fn(),
  writeFile: vi.fn().mockResolvedValue(undefined),
  mkdir: vi.fn().mockResolvedValue(undefined),
  readFile: vi.fn(),
}));

vi.mock("openai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("openai")>();
  class FakeOpenAI {
    responses = { create: mocks.responsesCreate };
    images = { generate: mocks.imagesGenerate };
    embeddings = { create: mocks.embeddingsCreate };
    models = { list: mocks.modelsList };
    audio = {
      speech: { create: mocks.speechCreate },
      transcriptions: { create: mocks.transcriptionsCreate },
    };
  }
  return {
    ...actual,
    default: FakeOpenAI,
  };
});

vi.mock("node:fs/promises", () => ({
  writeFile: mocks.writeFile,
  mkdir: mocks.mkdir,
  readFile: mocks.readFile,
}));

import { APIConnectionError, APIError, OpenAIError } from "openai";
import {
  createEmbeddings,
  formatError,
  generateImage,
  generateSpeech,
  generateText,
  listModels,
  transcribeAudio,
} from "./tools.js";

beforeEach(() => {
  for (const mock of Object.values(mocks)) {
    mock.mockReset();
  }
});

describe("formatError", () => {
  it("includes the status code for an APIError", () => {
    const error = new APIError(
      429,
      { message: "Rate limited" },
      "Rate limited",
      undefined,
    );
    expect(formatError(error)).toBe("OpenAI API error (429): 429 Rate limited");
  });

  it("falls back to 'unknown status' when an APIError has no status", () => {
    const error = new APIConnectionError({ message: "Network down" });
    expect(formatError(error)).toBe(
      "OpenAI API error (unknown status): Network down",
    );
  });

  it("uses the message directly for other OpenAIErrors", () => {
    const error = new OpenAIError("Missing credentials");
    expect(formatError(error)).toBe("Missing credentials");
  });

  it("uses the message for a plain Error", () => {
    expect(formatError(new Error("boom"))).toBe("boom");
  });

  it("stringifies non-Error values", () => {
    expect(formatError("oops")).toBe("oops");
  });
});

describe("generateText", () => {
  it("returns the model's output text", async () => {
    mocks.responsesCreate.mockResolvedValueOnce({ output_text: "Hello there" });

    const result = await generateText({ prompt: "hi", model: "gpt-4o-mini" });

    expect(result).toEqual({
      content: [{ type: "text", text: "Hello there" }],
    });
  });

  it("falls back to a placeholder when output_text is empty", async () => {
    mocks.responsesCreate.mockResolvedValueOnce({ output_text: "" });

    const result = await generateText({ prompt: "hi", model: "gpt-4o-mini" });

    expect(result.content[0]).toEqual({
      type: "text",
      text: "No response content.",
    });
  });

  it("returns a formatted error result when the API call fails", async () => {
    mocks.responsesCreate.mockRejectedValueOnce(
      new OpenAIError("Missing credentials"),
    );

    const result = await generateText({ prompt: "hi", model: "gpt-4o-mini" });

    expect(result.isError).toBe(true);
    expect(result.content[0]).toEqual({
      type: "text",
      text: "Missing credentials",
    });
  });

  it("passes an image URL through as input_image content", async () => {
    mocks.responsesCreate.mockResolvedValueOnce({ output_text: "A cat" });

    await generateText({
      prompt: "what's in this image?",
      model: "gpt-4o-mini",
      images: ["https://example.com/cat.png"],
    });

    expect(mocks.responsesCreate).toHaveBeenCalledWith({
      model: "gpt-4o-mini",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "what's in this image?" },
            {
              type: "input_image",
              image_url: "https://example.com/cat.png",
              detail: "auto",
            },
          ],
        },
      ],
    });
  });

  it("reads a local image file and inlines it as a base64 data URL", async () => {
    mocks.readFile.mockResolvedValueOnce(Buffer.from("fake-bytes"));
    mocks.responsesCreate.mockResolvedValueOnce({ output_text: "A cat" });

    await generateText({
      prompt: "what's in this image?",
      model: "gpt-4o-mini",
      images: ["C:/images/cat.png"],
    });

    expect(mocks.readFile).toHaveBeenCalledWith("C:/images/cat.png");
    const call = mocks.responsesCreate.mock.calls[0][0];
    expect(call.input[0].content[1]).toEqual({
      type: "input_image",
      image_url: `data:image/png;base64,${Buffer.from("fake-bytes").toString("base64")}`,
      detail: "auto",
    });
  });

  it("returns a formatted error for an unsupported local image extension", async () => {
    const result = await generateText({
      prompt: "what's in this image?",
      model: "gpt-4o-mini",
      images: ["C:/images/cat.bmp"],
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].type).toBe("text");
    expect((result.content[0] as { text: string }).text).toMatch(
      /Unsupported image file extension/,
    );
    expect(mocks.responsesCreate).not.toHaveBeenCalled();
  });
});

describe("generateImage", () => {
  it("writes base64 results to a temp file and returns its path by default", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({
      output_format: "png",
      data: [{ b64_json: "abc123" }],
    });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 1,
      returnAs: "path",
    });

    expect(mocks.writeFile).toHaveBeenCalledTimes(1);
    const [filePath, buffer] = mocks.writeFile.mock.calls[0];
    expect(filePath).toMatch(/openai-image-.+\.png$/);
    expect(Buffer.from(buffer).equals(Buffer.from("abc123", "base64"))).toBe(
      true,
    );
    expect(result.content).toEqual([{ type: "text", text: filePath }]);
  });

  it("saves to the given outputPath when provided", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({
      output_format: "png",
      data: [{ b64_json: "abc123" }],
    });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 1,
      returnAs: "path",
      outputPath: "C:/images/cat.png",
    });

    expect(mocks.mkdir).toHaveBeenCalledWith("C:/images", {
      recursive: true,
    });
    expect(mocks.writeFile).toHaveBeenCalledWith(
      "C:/images/cat.png",
      expect.anything(),
    );
    expect(result.content).toEqual([
      { type: "text", text: "C:/images/cat.png" },
    ]);
  });

  it("inserts an index into outputPath when generating multiple images", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({
      output_format: "png",
      data: [{ b64_json: "abc123" }, { b64_json: "def456" }],
    });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 2,
      returnAs: "path",
      outputPath: "C:/images/cat.png",
    });

    expect(result.content).toEqual([
      { type: "text", text: expect.stringMatching(/cat-1\.png$/) },
      { type: "text", text: expect.stringMatching(/cat-2\.png$/) },
    ]);
  });

  it("returns inline image content when returnAs is base64", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({
      output_format: "png",
      data: [{ b64_json: "abc123" }],
    });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 1,
      returnAs: "base64",
    });

    expect(mocks.writeFile).not.toHaveBeenCalled();
    expect(result.content).toEqual([
      { type: "image", data: "abc123", mimeType: "image/png" },
    ]);
  });

  it("falls back to a URL when no base64 data is present", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({
      data: [{ url: "https://example.com/cat.png" }],
    });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 1,
      returnAs: "path",
    });

    expect(result.content).toEqual([
      { type: "text", text: "https://example.com/cat.png" },
    ]);
  });

  it("reports when no images are returned", async () => {
    mocks.imagesGenerate.mockResolvedValueOnce({ data: [] });

    const result = await generateImage({
      prompt: "a cat",
      model: "gpt-image-1",
      n: 1,
      returnAs: "path",
    });

    expect(result.content).toEqual([
      { type: "text", text: "No images were returned." },
    ]);
  });
});

describe("createEmbeddings", () => {
  it("summarizes embedding vectors", async () => {
    mocks.embeddingsCreate.mockResolvedValueOnce({
      data: [{ index: 0, embedding: [0.1, 0.2] }],
    });

    const result = await createEmbeddings({
      input: "hello",
      model: "text-embedding-3-small",
    });

    expect(result.content[0]).toEqual({
      type: "text",
      text: JSON.stringify([{ index: 0, embedding: [0.1, 0.2] }]),
    });
  });
});

describe("listModels", () => {
  it("returns a sorted list of model ids", async () => {
    mocks.modelsList.mockReturnValueOnce(
      (async function* () {
        yield { id: "gpt-4o-mini" };
        yield { id: "gpt-3.5-turbo" };
      })(),
    );

    const result = await listModels();

    expect(result.content[0]).toEqual({
      type: "text",
      text: "gpt-3.5-turbo\ngpt-4o-mini",
    });
  });

  it("reports when no models are available", async () => {
    mocks.modelsList.mockReturnValueOnce((async function* () {})());

    const result = await listModels();

    expect(result.content[0]).toEqual({
      type: "text",
      text: "No models available.",
    });
  });
});

describe("generateSpeech", () => {
  it("writes audio to a temp file and returns its path by default", async () => {
    mocks.speechCreate.mockResolvedValueOnce({
      arrayBuffer: async () =>
        new TextEncoder().encode("fake-audio-bytes").buffer,
    });

    const result = await generateSpeech({
      input: "hello",
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      format: "mp3",
      returnAs: "path",
    });

    expect(mocks.writeFile).toHaveBeenCalledTimes(1);
    const [filePath, buffer] = mocks.writeFile.mock.calls[0];
    expect(filePath).toMatch(/openai-speech-.+\.mp3$/);
    expect(Buffer.from(buffer).toString()).toBe("fake-audio-bytes");
    expect(result.content).toEqual([{ type: "text", text: filePath }]);
  });

  it("saves to the given outputPath when provided", async () => {
    mocks.speechCreate.mockResolvedValueOnce({
      arrayBuffer: async () =>
        new TextEncoder().encode("fake-audio-bytes").buffer,
    });

    const result = await generateSpeech({
      input: "hello",
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      format: "mp3",
      returnAs: "path",
      outputPath: "C:/audio/greeting.mp3",
    });

    expect(mocks.mkdir).toHaveBeenCalledWith("C:/audio", { recursive: true });
    expect(result.content).toEqual([
      { type: "text", text: "C:/audio/greeting.mp3" },
    ]);
  });

  it("returns inline audio content when returnAs is base64", async () => {
    mocks.speechCreate.mockResolvedValueOnce({
      arrayBuffer: async () =>
        new TextEncoder().encode("fake-audio-bytes").buffer,
    });

    const result = await generateSpeech({
      input: "hello",
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      format: "wav",
      returnAs: "base64",
    });

    expect(mocks.writeFile).not.toHaveBeenCalled();
    expect(result.content).toEqual([
      {
        type: "audio",
        data: Buffer.from("fake-audio-bytes").toString("base64"),
        mimeType: "audio/wav",
      },
    ]);
  });

  it("returns a formatted error result when the API call fails", async () => {
    mocks.speechCreate.mockRejectedValueOnce(
      new OpenAIError("Missing credentials"),
    );

    const result = await generateSpeech({
      input: "hello",
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      format: "mp3",
      returnAs: "path",
    });

    expect(result.isError).toBe(true);
    expect(result.content[0]).toEqual({
      type: "text",
      text: "Missing credentials",
    });
  });
});

describe("transcribeAudio", () => {
  it("reads a local audio file and returns the transcribed text", async () => {
    mocks.readFile.mockResolvedValueOnce(Buffer.from("fake-audio-bytes"));
    mocks.transcriptionsCreate.mockResolvedValueOnce({ text: "hello world" });

    const result = await transcribeAudio({
      file: "C:/audio/clip.mp3",
      model: "gpt-4o-mini-transcribe",
      format: "json",
    });

    expect(mocks.readFile).toHaveBeenCalledWith("C:/audio/clip.mp3");
    expect(result.content).toEqual([{ type: "text", text: "hello world" }]);
  });

  it("downloads a remote URL and transcribes it", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      arrayBuffer: async () =>
        new TextEncoder().encode("fake-audio-bytes").buffer,
    } as Response);
    mocks.transcriptionsCreate.mockResolvedValueOnce({
      text: "hello from url",
    });

    const result = await transcribeAudio({
      file: "https://example.com/clip.mp3",
      model: "gpt-4o-mini-transcribe",
      format: "json",
    });

    expect(fetchMock).toHaveBeenCalledWith("https://example.com/clip.mp3");
    expect(result.content).toEqual([{ type: "text", text: "hello from url" }]);
    fetchMock.mockRestore();
  });

  it("returns the full JSON payload for verbose_json format", async () => {
    mocks.readFile.mockResolvedValueOnce(Buffer.from("fake-audio-bytes"));
    const verbose = {
      text: "hi",
      duration: 1.2,
      language: "en",
      segments: [],
    };
    mocks.transcriptionsCreate.mockResolvedValueOnce(verbose);

    const result = await transcribeAudio({
      file: "C:/audio/clip.mp3",
      model: "gpt-4o-mini-transcribe",
      format: "verbose_json",
    });

    expect(result.content).toEqual([
      { type: "text", text: JSON.stringify(verbose) },
    ]);
  });

  it("returns a plain string response as-is for text format", async () => {
    mocks.readFile.mockResolvedValueOnce(Buffer.from("fake-audio-bytes"));
    mocks.transcriptionsCreate.mockResolvedValueOnce("hello world");

    const result = await transcribeAudio({
      file: "C:/audio/clip.mp3",
      model: "gpt-4o-mini-transcribe",
      format: "text",
    });

    expect(result.content).toEqual([{ type: "text", text: "hello world" }]);
  });

  it("returns a formatted error when a remote download fails", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: "Not Found",
    } as Response);

    const result = await transcribeAudio({
      file: "https://example.com/missing.mp3",
      model: "gpt-4o-mini-transcribe",
      format: "json",
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].type).toBe("text");
    expect((result.content[0] as { text: string }).text).toMatch(
      /Failed to download audio/,
    );
    expect(mocks.transcriptionsCreate).not.toHaveBeenCalled();
    fetchMock.mockRestore();
  });
});
