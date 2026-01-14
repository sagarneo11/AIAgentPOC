import { POST } from "@/app/api/chat/route";
import { NextRequest } from "next/server";

// Mock dependencies
jest.mock("@langchain/community/chat_models/ollama", () => ({
  ChatOllama: jest.fn(),
}));

jest.mock("@/lib/vector-store", () => ({
  getVectorStore: jest.fn(),
}));

describe("/api/chat", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should return 400 if query is missing", async () => {
    const request = new NextRequest("http://localhost:3000/api/chat", {
      method: "POST",
      body: JSON.stringify({ messages: [] }),
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data.error).toBe("Query is required");
  });

  it("should return 400 if query is empty", async () => {
    const request = new NextRequest("http://localhost:3000/api/chat", {
      method: "POST",
      body: JSON.stringify({ messages: [], query: "" }),
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data.error).toBe("Query is required");
  });

  it("should handle chat request with valid query", async () => {
    const { getVectorStore } = require("@/lib/vector-store");
    
    getVectorStore.mockResolvedValue({
      asRetriever: jest.fn().mockReturnValue({
        getRelevantDocuments: jest.fn().mockResolvedValue([]),
      }),
    });

    const request = new NextRequest("http://localhost:3000/api/chat", {
      method: "POST",
      body: JSON.stringify({
        messages: [],
        query: "What is the main topic?",
      }),
    });

    // Note: This test may need adjustment based on actual streaming implementation
    const response = await POST(request);
    
    // Should return a streaming response
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
  });
});
