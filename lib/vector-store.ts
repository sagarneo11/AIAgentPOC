import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

// Global singleton vector store that persists across API calls
let vectorStore: MemoryVectorStore | null = null;
let embeddings: OllamaEmbeddings | null = null;

export async function getVectorStore(): Promise<MemoryVectorStore> {
  if (!embeddings) {
    embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      baseUrl: "http://localhost:11434",
    });
  }

  if (!vectorStore) {
    vectorStore = new MemoryVectorStore(embeddings);
  }

  return vectorStore;
}

export function resetVectorStore() {
  vectorStore = null;
}

export function hasVectorStore(): boolean {
  return vectorStore !== null;
}
