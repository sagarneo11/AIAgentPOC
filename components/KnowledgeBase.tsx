"use client";

import { useState, useEffect } from "react";
import { Upload, Link, CheckCircle2, Loader2, FileText, Globe, RefreshCw, Database, Brain, Trash2, AlertTriangle } from "lucide-react";
import FineTuningPanel from "./FineTuningPanel";

interface Document {
  id: number;
  type: string;
  source: string;
  filename?: string;
  chunks_count: number;
  indexed_at: string;
  fine_tuned_model?: string;
}

interface KnowledgeBaseProps {
  selectedDocId: number | null;
  onDocSelect: (docId: number | null) => void;
  selectedFineTunedModel: string | null;
  onFineTunedModelSelect: (modelName: string | null) => void;
}

export default function KnowledgeBase({ 
  selectedDocId, 
  onDocSelect,
  selectedFineTunedModel,
  onFineTunedModelSelect
}: KnowledgeBaseProps) {
  const [activeTab, setActiveTab] = useState<"index" | "fine-tune">("index");
  const [url, setUrl] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isIndexed, setIsIndexed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("url", url);

      const response = await fetch("http://localhost:8000/api/ingest", {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(`API returned non-JSON response: ${text.substring(0, 100)}`);
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to index URL");
      }

      setIsIndexed(true);
      setUrl("");
      fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to index URL");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/api/ingest", {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(`API returned non-JSON response: ${text.substring(0, 100)}`);
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to index PDF");
      }

      setIsIndexed(true);
      e.target.value = "";
      fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to index PDF");
    } finally {
      setIsUploading(false);
    }
  };

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

  const handleDeleteDocument = async (docId: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent document selection when clicking delete
    
    if (!confirm(`Are you sure you want to delete this document? This will remove all its chunks from the vector store.`)) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/documents/${docId}`, {
        method: "DELETE",
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "Failed to delete document");
      }

      // If the deleted document was selected, deselect it
      if (selectedDocId === docId) {
        onDocSelect(null);
      }

      // Refresh the document list
      fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete document");
    }
  };

  const handleClearAll = async () => {
    if (!confirm(`⚠️ WARNING: This will permanently delete ALL data:\n\n- All indexed documents\n- All vector store data\n- All training data\n- All fine-tuned modelfiles\n\nThis action cannot be undone!\n\nAre you absolutely sure?`)) {
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/api/clear-all", {
        method: "DELETE",
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "Failed to clear all data");
      }

      // Clear selections
      onDocSelect(null);
      onFineTunedModelSelect(null);

      // Refresh the document list
      fetchDocuments();
      
      // Show success message
      setIsIndexed(false);
      setError(null);
      alert("✅ All data cleared successfully!");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear all data");
    }
  };

  useEffect(() => {
    fetchDocuments();
    // Refresh documents periodically
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800">
      <div className="p-6 border-b border-gray-200 dark:border-gray-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Knowledge Base
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Manage your documents and fine-tuned models
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-800">
        <button
          onClick={() => setActiveTab("index")}
          className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
            activeTab === "index"
              ? "text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20"
              : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <Database className="w-4 h-4" />
            Index Documents
          </div>
        </button>
        <button
          onClick={() => setActiveTab("fine-tune")}
          className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
            activeTab === "fine-tune"
              ? "text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20"
              : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <Brain className="w-4 h-4" />
            Fine-Tuning
          </div>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeTab === "index" ? (
          <div className="p-6 space-y-6">
            {/* URL Input - Hidden */}
            {/* <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Link className="inline-block w-4 h-4 mr-2" />
                Enter URL
              </label>
              <form onSubmit={handleUrlSubmit} className="space-y-2">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isUploading}
                />
                <button
                  type="submit"
                  disabled={isUploading || !url.trim()}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Indexing...
                    </>
                  ) : (
                    "Index URL"
                  )}
                </button>
              </form>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300 dark:border-gray-700"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-gray-50 dark:bg-gray-900 text-gray-500">
                  OR
                </span>
              </div>
            </div> */}

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Upload className="inline-block w-4 h-4 mr-2" />
                Upload PDF
              </label>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 dark:border-gray-700 border-dashed rounded-lg cursor-pointer bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  {isUploading ? (
                    <Loader2 className="w-8 h-8 mb-2 text-gray-400 animate-spin" />
                  ) : (
                    <Upload className="w-8 h-8 mb-2 text-gray-400" />
                  )}
                  <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                    {isUploading ? (
                      <span>Indexing PDF...</span>
                    ) : (
                      <span className="font-semibold">Click to upload</span>
                    )}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    PDF only
                  </p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept="application/pdf"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
              </label>
            </div>

            {/* Success Indicator */}
            {isIndexed && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
                <span className="text-sm text-green-800 dark:text-green-300">
                  Document Indexed Successfully
                </span>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
              </div>
            )}

            {/* Indexed Documents List */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Indexed Documents
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={fetchDocuments}
                    disabled={isLoadingDocs}
                    className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50"
                    title="Refresh list"
                  >
                    <RefreshCw className={`w-4 h-4 ${isLoadingDocs ? "animate-spin" : ""}`} />
                  </button>
                  {documents.length > 0 && (
                    <button
                      onClick={handleClearAll}
                      className="p-1.5 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                      title="Clear all data"
                    >
                      <AlertTriangle className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>

              {isLoadingDocs ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
                </div>
              ) : documents.length === 0 ? (
                <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    No documents indexed yet
                  </p>
                </div>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      onClick={() => onDocSelect(selectedDocId === doc.id ? null : doc.id)}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedDocId === doc.id
                          ? "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700"
                          : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {doc.type === "pdf" ? (
                          <FileText className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        ) : (
                          <Globe className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                            {doc.filename || doc.source}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            {doc.chunks_count} chunks • {new Date(doc.indexed_at).toLocaleDateString()}
                          </p>
                          {doc.fine_tuned_model && (
                            <p className="text-xs text-purple-600 dark:text-purple-400 mt-1 flex items-center gap-1">
                              <Brain className="w-3 h-3" />
                              Fine-tuned: {doc.fine_tuned_model}
                            </p>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          {selectedDocId === doc.id && (
                            <CheckCircle2 className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                          )}
                          <button
                            onClick={(e) => handleDeleteDocument(doc.id, e)}
                            className="p-1.5 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                            title="Delete document"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <FineTuningPanel 
            documents={documents}
            onDocumentsUpdate={fetchDocuments}
            selectedFineTunedModel={selectedFineTunedModel}
            onFineTunedModelSelect={onFineTunedModelSelect}
          />
        )}
      </div>
    </div>
  );
}
