import { NextRequest, NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { getVectorStore } from "@/lib/vector-store";
import { extractTextFromPDF, extractTextFromURL } from "@/lib/text-extraction";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function GET() {
  return NextResponse.json({ message: "Ingest API is working" });
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const url = formData.get("url") as string | null;

    if (!file && !url) {
      return NextResponse.json(
        { error: "Either file or URL must be provided" },
        { status: 400 }
      );
    }

    let text: string;

    if (file) {
      if (file.type !== "application/pdf") {
        return NextResponse.json(
          { error: "Only PDF files are supported" },
          { status: 400 }
        );
      }

      const buffer = Buffer.from(await file.arrayBuffer());
      text = await extractTextFromPDF(buffer);
    } else if (url) {
      text = await extractTextFromURL(url);
    } else {
      return NextResponse.json(
        { error: "Invalid input" },
        { status: 400 }
      );
    }

    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { error: "No text content extracted" },
        { status: 400 }
      );
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await splitter.createDocuments([text]);

    const vectorStore = await getVectorStore();
    await vectorStore.addDocuments(chunks);

    return NextResponse.json({
      success: true,
      message: "Document indexed successfully",
      chunksCount: chunks.length,
    });
  } catch (error) {
    console.error("Ingest error:", error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to ingest document",
      },
      { status: 500 }
    );
  }
}
