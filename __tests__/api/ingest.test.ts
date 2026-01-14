import { POST } from "@/app/api/ingest/route";
import { NextRequest } from "next/server";
import { resetVectorStore } from "@/lib/vector-store";

// Mock dependencies
jest.mock("pdf-parse", () => ({
  __esModule: true,
  default: jest.fn(),
}));

jest.mock("cheerio", () => ({
  load: jest.fn(),
}));

jest.mock("@/lib/vector-store", () => ({
  getVectorStore: jest.fn(),
  resetVectorStore: jest.fn(),
}));

jest.mock("@/lib/text-extraction", () => ({
  extractTextFromPDF: jest.fn(),
  extractTextFromURL: jest.fn(),
}));

describe("/api/ingest", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    resetVectorStore();
  });

  it("should return 400 if neither file nor URL is provided", async () => {
    const formData = new FormData();
    const request = new NextRequest("http://localhost:3000/api/ingest", {
      method: "POST",
      body: formData,
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data.error).toBe("Either file or URL must be provided");
  });

  it("should return 400 if file is not a PDF", async () => {
    const formData = new FormData();
    const file = new File(["test"], "test.txt", { type: "text/plain" });
    formData.append("file", file);

    const request = new NextRequest("http://localhost:3000/api/ingest", {
      method: "POST",
      body: formData,
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data.error).toBe("Only PDF files are supported");
  });

  it("should handle PDF file upload successfully", async () => {
    const { extractTextFromPDF } = require("@/lib/text-extraction");
    const { getVectorStore } = require("@/lib/vector-store");

    extractTextFromPDF.mockResolvedValue("Test PDF content");
    getVectorStore.mockResolvedValue({
      addDocuments: jest.fn().mockResolvedValue(undefined),
    });

    const formData = new FormData();
    const file = new File(["test"], "test.pdf", { type: "application/pdf" });
    formData.append("file", file);

    const request = new NextRequest("http://localhost:3000/api/ingest", {
      method: "POST",
      body: formData,
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.message).toBe("Document indexed successfully");
  });

  it("should handle URL submission successfully", async () => {
    const { extractTextFromURL } = require("@/lib/text-extraction");
    const { getVectorStore } = require("@/lib/vector-store");

    extractTextFromURL.mockResolvedValue("Test URL content");
    getVectorStore.mockResolvedValue({
      addDocuments: jest.fn().mockResolvedValue(undefined),
    });

    const formData = new FormData();
    formData.append("url", "https://example.com");

    const request = new NextRequest("http://localhost:3000/api/ingest", {
      method: "POST",
      body: formData,
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
  });
});
