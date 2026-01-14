import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ChatPanel from "@/components/ChatPanel";

// Mock fetch
global.fetch = jest.fn();

// Mock react-markdown
jest.mock("react-markdown", () => ({
  __esModule: true,
  default: ({ children }: { children: string }) => <div>{children}</div>,
}));

describe("ChatPanel Component", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  it("renders chat panel title and description", () => {
    render(<ChatPanel />);
    
    expect(screen.getByText("Agent Chat")).toBeInTheDocument();
    expect(screen.getByText(/Ask questions about your indexed documents/)).toBeInTheDocument();
  });

  it("renders input field and send button", () => {
    render(<ChatPanel />);
    
    const input = screen.getByPlaceholderText("Ask a question...");
    const sendButton = screen.getByRole("button", { name: /send/i });

    expect(input).toBeInTheDocument();
    expect(sendButton).toBeInTheDocument();
  });

  it("displays empty state message when no messages", () => {
    render(<ChatPanel />);
    
    expect(screen.getByText("Start a conversation by asking a question")).toBeInTheDocument();
  });

  it("handles message submission", async () => {
    // Mock streaming response
    const mockReader = {
      read: jest.fn()
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"content":"Hello"}\n\n'),
        })
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"content":" World"}\n\n'),
        })
        .mockResolvedValueOnce({ done: true }),
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    });

    render(<ChatPanel />);
    
    const input = screen.getByPlaceholderText("Ask a question...");
    const form = input.closest("form");

    fireEvent.change(input, { target: { value: "Hello" } });
    fireEvent.submit(form!);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith("/api/chat", expect.any(Object));
    });
  });

  it("displays user message after submission", async () => {
    const mockReader = {
      read: jest.fn().mockResolvedValueOnce({ done: true }),
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    });

    render(<ChatPanel />);
    
    const input = screen.getByPlaceholderText("Ask a question...");
    const form = input.closest("form");

    fireEvent.change(input, { target: { value: "Test question" } });
    fireEvent.submit(form!);

    await waitFor(() => {
      expect(screen.getByText("Test question")).toBeInTheDocument();
    });
  });

  it("displays loading state while processing", async () => {
    const mockReader = {
      read: jest.fn().mockImplementation(() => new Promise(() => {})), // Never resolves
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    });

    render(<ChatPanel />);
    
    const input = screen.getByPlaceholderText("Ask a question...");
    const form = input.closest("form");

    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(form!);

    await waitFor(() => {
      expect(screen.getByText("Thinking...")).toBeInTheDocument();
    });
  });

  it("displays error message on API failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: "Failed to get response" }),
    });

    render(<ChatPanel />);
    
    const input = screen.getByPlaceholderText("Ask a question...");
    const form = input.closest("form");

    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(form!);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
