import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import KnowledgeBase from "@/components/KnowledgeBase";

// Mock fetch
global.fetch = jest.fn();

describe("KnowledgeBase Component", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  it("renders knowledge base title and description", () => {
    render(<KnowledgeBase />);
    
    expect(screen.getByText("Knowledge Base")).toBeInTheDocument();
    expect(screen.getByText(/Upload a PDF or enter a URL to index/)).toBeInTheDocument();
  });

  it("renders URL input field", () => {
    render(<KnowledgeBase />);
    
    const urlInput = screen.getByPlaceholderText("https://example.com/article");
    expect(urlInput).toBeInTheDocument();
  });

  it("renders file upload area", () => {
    render(<KnowledgeBase />);
    
    expect(screen.getByText("Click to upload")).toBeInTheDocument();
    expect(screen.getByText("PDF only")).toBeInTheDocument();
  });

  it("handles URL submission", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Document indexed successfully" }),
    });

    render(<KnowledgeBase />);
    
    const urlInput = screen.getByPlaceholderText("https://example.com/article");
    const submitButton = screen.getByText("Index URL");

    fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith("/api/ingest", expect.any(Object));
    });
  });

  it("handles file upload", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Document indexed successfully" }),
    });

    render(<KnowledgeBase />);
    
    const fileInput = screen.getByLabelText(/Upload PDF/i) as HTMLInputElement;
    const file = new File(["test"], "test.pdf", { type: "application/pdf" });

    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });
  });

  it("displays success message after successful indexing", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Document indexed successfully" }),
    });

    render(<KnowledgeBase />);
    
    const urlInput = screen.getByPlaceholderText("https://example.com/article");
    const submitButton = screen.getByText("Index URL");

    fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText("Document Indexed Successfully")).toBeInTheDocument();
    });
  });

  it("displays error message on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: "Failed to index URL" }),
    });

    render(<KnowledgeBase />);
    
    const urlInput = screen.getByPlaceholderText("https://example.com/article");
    const submitButton = screen.getByText("Index URL");

    fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText("Failed to index URL")).toBeInTheDocument();
    });
  });
});
