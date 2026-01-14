"use client";

import { useState, useEffect } from "react";
import { Brain, Loader2, CheckCircle2, FileText, Play, Settings, Sparkles, Trash2, X } from "lucide-react";

interface Document {
  id: number;
  type: string;
  source: string;
  filename?: string;
  chunks_count: number;
  indexed_at: string;
  fine_tuned_model?: string;
}

interface TrainingStatus {
  training_data_prepared: boolean;
  model_trained: boolean;
  documents_count: number;
  total_chunks: number;
}

interface FineTuningPanelProps {
  documents: Document[];
  onDocumentsUpdate: () => void;
  selectedFineTunedModel: string | null;
  onFineTunedModelSelect: (modelName: string | null) => void;
}

export default function FineTuningPanel({ 
  documents, 
  onDocumentsUpdate,
  selectedFineTunedModel,
  onFineTunedModelSelect
}: FineTuningPanelProps) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [isCreatingModelfile, setIsCreatingModelfile] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fineTunedModels, setFineTunedModels] = useState<Record<number, string>>({});
  const [trainingDataFiles, setTrainingDataFiles] = useState<Array<{filename: string, size: number, modified: number}>>([]);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/fine-tune/status");
      const data = await response.json();
      if (data.success) {
        setStatus(data);
      }
      
      // Fetch fine-tuned models for each document
      const modelsResponse = await fetch("http://localhost:8000/api/fine-tune/models");
      if (modelsResponse.ok) {
        const modelsData = await modelsResponse.json();
        if (modelsData.success) {
          setFineTunedModels(modelsData.models || {});
        }
      }
      
      // Fetch training data files
      const filesResponse = await fetch("http://localhost:8000/api/fine-tune/training-data");
      if (filesResponse.ok) {
        const filesData = await filesResponse.json();
        if (filesData.success) {
          setTrainingDataFiles(filesData.files || []);
        }
      }
    } catch (err) {
      console.error("Failed to fetch status:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handlePrepareTrainingData = async (docId: number | null = null) => {
    setIsPreparing(true);
    setError(null);
    setMessage(null);

    try {
      const url = docId 
        ? `http://localhost:8000/api/fine-tune/prepare?format=alpaca&doc_id=${docId}`
        : `http://localhost:8000/api/fine-tune/prepare?format=alpaca`;
      
      const response = await fetch(url, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setMessage(`Training data prepared: ${data.output_file}`);
        if (docId) {
          setSelectedDocId(docId);
        }
        fetchStatus();
      } else {
        setError(data.detail || "Failed to prepare training data");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to prepare training data");
    } finally {
      setIsPreparing(false);
    }
  };

  const handleCreateModelfile = async (docId: number | null = null) => {
    setIsCreatingModelfile(true);
    setError(null);
    setMessage(null);

    try {
      const url = docId
        ? `http://localhost:8000/api/fine-tune/create-modelfile?base_model=phi3:mini&doc_id=${docId}`
        : `http://localhost:8000/api/fine-tune/create-modelfile?base_model=phi3:mini`;

      const response = await fetch(url, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        const modelName = data.model_name || "rag-smolvlm";
        setMessage(
          `Modelfile created! Run: ollama create ${modelName} -f ${data.modelfile_path}`
        );
        if (docId) {
          setSelectedDocId(docId);
        }
        fetchStatus();
        onDocumentsUpdate();
      } else {
        setError(data.detail || "Failed to create Modelfile");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create Modelfile");
    } finally {
      setIsCreatingModelfile(false);
    }
  };

  const handleDeleteTrainingData = async (filename: string) => {
    setIsDeleting(filename);
    setError(null);
    setMessage(null);

    try {
      const response = await fetch(`http://localhost:8000/api/fine-tune/training-data/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (data.success) {
        setMessage(`Training data file ${filename} deleted successfully`);
        setShowDeleteConfirm(null);
        fetchStatus(); // Refresh the list
      } else {
        setError(data.detail || "Failed to delete training data");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete training data");
    } finally {
      setIsDeleting(null);
    }
  };

  const handleDeleteAllTrainingData = async () => {
    setIsDeleting("all");
    setError(null);
    setMessage(null);

    try {
      const response = await fetch("http://localhost:8000/api/fine-tune/training-data", {
        method: "DELETE",
      });

      const data = await response.json();

      if (data.success) {
        setMessage(`Deleted ${data.count} training data file(s)`);
        setShowDeleteConfirm(null);
        fetchStatus(); // Refresh the list
      } else {
        setError(data.detail || "Failed to delete training data");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete training data");
    } finally {
      setIsDeleting(null);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Fine-Tuning
        </h3>
      </div>

      {/* Fine-Tuned Models Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Fine-Tuned Model
        </label>
        <div className="space-y-2">
          <button
            onClick={() => onFineTunedModelSelect(null)}
            className={`w-full p-3 text-left border rounded-lg transition-colors ${
              selectedFineTunedModel === null
                ? "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700"
                : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Base Model (phi3:mini)
              </span>
              {selectedFineTunedModel === null && (
                <CheckCircle2 className="w-4 h-4 text-blue-500" />
              )}
            </div>
          </button>

          {Object.entries(fineTunedModels).map(([docId, modelName]) => {
            const doc = documents.find(d => d.id === parseInt(docId));
            return (
              <button
                key={docId}
                onClick={() => onFineTunedModelSelect(modelName)}
                className={`w-full p-3 text-left border rounded-lg transition-colors ${
                  selectedFineTunedModel === modelName
                    ? "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700"
                    : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100 block">
                      {modelName}
                    </span>
                    {doc && (
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {doc.filename || doc.source}
                      </span>
                    )}
                  </div>
                  {selectedFineTunedModel === modelName && (
                    <CheckCircle2 className="w-4 h-4 text-blue-500" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Document Selection for Fine-Tuning */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Document to Fine-Tune
        </label>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => setSelectedDocId(selectedDocId === doc.id ? null : doc.id)}
              className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                selectedDocId === doc.id
                  ? "bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700"
                  : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {doc.filename || doc.source}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {doc.chunks_count} chunks
                    </p>
                  </div>
                </div>
                {doc.fine_tuned_model && (
                  <Sparkles className="w-4 h-4 text-purple-500" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Status */}
      {isLoading ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
        </div>
      ) : status ? (
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-xs text-gray-500 dark:text-gray-400">Documents</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {status.documents_count}
            </p>
          </div>
          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-xs text-gray-500 dark:text-gray-400">Total Chunks</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {status.total_chunks}
            </p>
          </div>
        </div>
      ) : null}

      {/* Actions */}
      <div className="space-y-2 pt-2">
        <button
          onClick={() => handlePrepareTrainingData(selectedDocId)}
          disabled={isPreparing || (selectedDocId === null && status?.documents_count === 0)}
          className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isPreparing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Preparing...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Prepare Training Data {selectedDocId ? `(Doc ${selectedDocId})` : "(All)"}
            </>
          )}
        </button>

        <button
          onClick={() => handleCreateModelfile(selectedDocId)}
          disabled={isCreatingModelfile || (selectedDocId === null && status?.documents_count === 0)}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isCreatingModelfile ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Settings className="w-4 h-4" />
              Create Ollama Modelfile {selectedDocId ? `(Doc ${selectedDocId})` : "(All)"}
            </>
          )}
        </button>
      </div>

      {message && (
        <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <p className="text-sm text-green-800 dark:text-green-300">{message}</p>
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Training Data Files List */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Training Data Files
          </label>
          {trainingDataFiles.length > 0 && (
            <button
              onClick={() => setShowDeleteConfirm("all")}
              className="text-xs text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
            >
              Delete All
            </button>
          )}
        </div>

        {trainingDataFiles.length === 0 ? (
          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              No training data files
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {trainingDataFiles.map((file) => (
              <div
                key={file.filename}
                className="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg flex items-center justify-between"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {file.filename}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {formatFileSize(file.size)} • {new Date(file.modified * 1000).toLocaleString()}
                  </p>
                </div>
                <button
                  onClick={() => setShowDeleteConfirm(file.filename)}
                  disabled={isDeleting === file.filename}
                  className="ml-2 p-1.5 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded disabled:opacity-50"
                  title="Delete file"
                >
                  {isDeleting === file.filename ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">
              Confirm Delete
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
              {showDeleteConfirm === "all"
                ? `Are you sure you want to delete all ${trainingDataFiles.length} training data file(s)? This action cannot be undone.`
                : `Are you sure you want to delete "${showDeleteConfirm}"? This action cannot be undone.`}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(null)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (showDeleteConfirm === "all") {
                    handleDeleteAllTrainingData();
                  } else {
                    handleDeleteTrainingData(showDeleteConfirm);
                  }
                }}
                disabled={isDeleting !== null}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isDeleting && (
                  <Loader2 className="w-4 h-4 animate-spin" />
                )}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
