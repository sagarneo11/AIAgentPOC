"use client";

import { useState } from "react";
import KnowledgeBase from "@/components/KnowledgeBase";
import ChatPanel from "@/components/ChatPanel";

export default function Home() {
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [selectedFineTunedModel, setSelectedFineTunedModel] = useState<string | null>(null);

  return (
    <main className="h-screen flex">
      {/* Left Sidebar - Knowledge Base (30%) */}
      <div className="w-[30%] min-w-[300px]">
        <KnowledgeBase 
          selectedDocId={selectedDocId}
          onDocSelect={setSelectedDocId}
          selectedFineTunedModel={selectedFineTunedModel}
          onFineTunedModelSelect={setSelectedFineTunedModel}
        />
      </div>

      {/* Right Panel - Chat (70%) */}
      <div className="flex-1">
        <ChatPanel 
          selectedDocId={selectedDocId}
          selectedFineTunedModel={selectedFineTunedModel}
          onDocSelect={setSelectedDocId}
        />
      </div>
    </main>
  );
}
