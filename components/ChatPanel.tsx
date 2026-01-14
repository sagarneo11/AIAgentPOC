"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Bot, User, FileText, Globe, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Document {
  id: number;
  type: string;
  source: string;
  filename?: string;
  chunks_count: number;
  indexed_at: string;
}

interface ChatPanelProps {
  selectedDocId: number | null;
  selectedFineTunedModel: string | null;
  onDocSelect?: (docId: number | null) => void;
}

export default function ChatPanel({ selectedDocId, selectedFineTunedModel, onDocSelect }: ChatPanelProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [showDocSelector, setShowDocSelector] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchDocuments = async () => {
    setIsLoadingDocs(true);
    try {
      const response = await fetch("http://localhost:8000/api/documents");
      const data = await response.json();
      if (data.success) {
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);

    // Add user message
    const newMessages: Message[] = [
      ...messages,
      { role: "user", content: userMessage },
    ];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: newMessages,
          query: userMessage,
          doc_id: selectedDocId,
          fine_tuned_model: selectedFineTunedModel,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to get response");
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content) {
                  assistantMessage += data.content;
                  setMessages([
                    ...newMessages,
                    { role: "assistant", content: assistantMessage },
                  ]);
                }
                if (data.error) {
                  throw new Error(data.error);
                }
              } catch (e) {
                // Skip invalid JSON
              }
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get response");
      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-950">
      <div className="p-6 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Agent Chat
          </h2>
          {/* Document Selector */}
          <div className="relative">
            <button
              onClick={() => setShowDocSelector(!showDocSelector)}
              className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2"
            >
              {selectedDocId ? (
                <>
                  {documents.find(d => d.id === selectedDocId)?.type === "pdf" ? (
                    <FileText className="w-4 h-4" />
                  ) : (
                    <Globe className="w-4 h-4" />
                  )}
                  <span className="max-w-[150px] truncate">
                    {documents.find(d => d.id === selectedDocId)?.filename || 
                     documents.find(d => d.id === selectedDocId)?.source || 
                     `Document ${selectedDocId}`}
                  </span>
                  {onDocSelect && (
                    <X 
                      className="w-3 h-3 hover:text-red-500" 
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onDocSelect) onDocSelect(null);
                      }}
                    />
                  )}
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4" />
                  <span>Select Document</span>
                </>
              )}
            </button>

            {showDocSelector && (
              <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-10 max-h-80 overflow-y-auto">
                <div className="p-2">
                  <button
                    onClick={() => {
                      if (onDocSelect) onDocSelect(null);
                      setShowDocSelector(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded text-sm ${
                      selectedDocId === null
                        ? "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
                        : "hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                    }`}
                  >
                    All Documents
                  </button>
                  {documents.map((doc) => (
                    <button
                      key={doc.id}
                      onClick={() => {
                        if (onDocSelect) onDocSelect(doc.id);
                        setShowDocSelector(false);
                      }}
                      className={`w-full text-left px-3 py-2 rounded text-sm flex items-center gap-2 ${
                        selectedDocId === doc.id
                          ? "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
                          : "hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                      }`}
                    >
                      {doc.type === "pdf" ? (
                        <FileText className="w-4 h-4 flex-shrink-0" />
                      ) : (
                        <Globe className="w-4 h-4 flex-shrink-0" />
                      )}
                      <span className="truncate">
                        {doc.filename || doc.source}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          {selectedFineTunedModel 
            ? `Using fine-tuned model: ${selectedFineTunedModel}`
            : selectedDocId 
            ? `Asking questions about: ${documents.find(d => d.id === selectedDocId)?.filename || documents.find(d => d.id === selectedDocId)?.source || "selected document"}`
            : "Ask questions about your indexed documents"}
        </p>
      </div>

      {/* Messages */}
      <div 
        className="flex-1 overflow-y-auto p-6 space-y-4"
        onClick={() => setShowDocSelector(false)}
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Start a conversation by asking a question</p>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-4 ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {message.role === "assistant" && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                <Bot className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
            )}

            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 ${
                message.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              }`}
            >
              {message.role === "user" ? (
                <p className="whitespace-pre-wrap">{message.content}</p>
              ) : (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {message.role === "user" && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                <User className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 justify-start">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
              <Bot className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-3 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
              <span className="text-gray-500 dark:text-gray-400">Thinking...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t border-gray-200 dark:border-gray-800">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
