import { NextRequest } from "next/server";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { getVectorStore } from "@/lib/vector-store";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ollama = new ChatOllama({
  model: "richardyoung/smolvlm2-2.2b-instruct",
  baseUrl: "http://localhost:11434",
  temperature: 0.7,
});

export async function POST(request: NextRequest) {
  try {
    const { messages, query } = await request.json();

    if (!query) {
      return new Response(
        JSON.stringify({ error: "Query is required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const vectorStore = await getVectorStore();

    // Check if vector store has documents by trying to retrieve
    // This will help catch cases where no documents are indexed
    const retriever = vectorStore.asRetriever({
      k: 4,
    });

    // Create the retrieval chain
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI assistant that answers questions based on the provided context. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Context: {context}`,
      ],
      ["human", "{input}"],
    ]);

    const documentChain = await createStuffDocumentsChain({
      llm: ollama,
      prompt,
    });

    const retrievalChain = await createRetrievalChain({
      combineDocsChain: documentChain,
      retriever,
    });

    // Convert messages to LangChain format
    const langchainMessages = messages.map((msg: any) => {
      if (msg.role === "user") {
        return new HumanMessage(msg.content);
      } else {
        return new AIMessage(msg.content);
      }
    });

    // Create a readable stream for the response
    const stream = new ReadableStream({
      async start(controller) {
        try {
          const result = await retrievalChain.invoke({
            input: query,
            chat_history: langchainMessages.slice(0, -1), // Exclude the current query
          });

          const answer = result.answer || "I couldn't generate a response.";

          // Stream the response in chunks
          const encoder = new TextEncoder();
          const chunks = answer.split(" ");
          
          for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i] + (i < chunks.length - 1 ? " " : "");
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: chunk })}\n\n`));
            // Small delay to simulate streaming
            await new Promise((resolve) => setTimeout(resolve, 20));
          }

          controller.close();
        } catch (error) {
          console.error("Chat error:", error);
          const encoder = new TextEncoder();
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ error: error instanceof Error ? error.message : "Failed to generate response" })}\n\n`
            )
          );
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Chat error:", error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Failed to process chat",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
