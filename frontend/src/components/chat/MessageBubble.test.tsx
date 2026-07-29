import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageBubble from "./MessageBubble";
import ChatPanel from "./ChatPanel";
import { useChatStore, type ChatMsg } from "@/store/chat-store";
import { api } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: {
    getValidAccessToken: vi.fn(),
    get: vi.fn(),
    patch: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    streamPost: vi.fn(),
  },
  API_BASE: "",
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => {
      const translations: Record<string, string> = {
        "chat.cancelOutput": "Stop response",
        "chat.sendMessage": "Send message",
        "chat.selectPlaceholder": "Select a document first...",
        "chat.routingMode": "Response mode",
        "chat.modeAuto": "Auto",
        "chat.modeQuick": "Quick",
        "chat.modeResearch": "Research",
      };
      return translations[key] || options?.defaultValue || key;
    },
    i18n: { language: "en" },
  }),
}));

const makeMessage = (overrides: Partial<ChatMsg>): ChatMsg => ({
  id: "msg-1",
  role: "assistant",
  content: "Assistant response",
  sources: [],
  ...overrides,
});

describe("MessageBubble", () => {
  it("renders a user message without assistant actions", () => {
    render(
      <MessageBubble
        message={makeMessage({
          role: "user",
          content: "Summarize my uploaded report",
        })}
      />,
    );

    expect(screen.getByText("Summarize my uploaded report")).toBeInTheDocument();
    expect(screen.queryByText("Was this helpful?")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Copiar respuesta")).not.toBeInTheDocument();
  });

  it("renders an assistant message with markdown and response controls", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content: "Here is the answer:\n\n```ts\nconst ok = true;\n```",
        })}
      />,
    );

    expect(screen.getByText("Here is the answer:")).toBeInTheDocument();
    expect(container.querySelector("pre")).toHaveTextContent("const ok = true;");
    expect(screen.getByText("Was this helpful?")).toBeInTheDocument();
    expect(screen.getByLabelText("Copiar respuesta")).toBeInTheDocument();
    expect(screen.getByLabelText("Compartir respuesta")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Leer respuesta")).toHaveLength(2);
  });

  it("renders standard and accented LaTeX as math", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content: String.raw`$\sqrt{3x-1}+(1+x)^2$ y $´\sqrt{3x-1}+(1+x)^2´$`,
        })}
      />,
    );

    expect(container.querySelectorAll(".katex")).toHaveLength(2);
    expect(container.querySelectorAll(".katex-error")).toHaveLength(0);
  });
});

describe("chat routing preference", () => {
  it("persists the selected mode and keeps it when chat state resets", () => {
    useChatStore.getState().setRoutingMode("research");
    useChatStore.getState().resetChat();

    expect(useChatStore.getState().routingMode).toBe("research");
    expect(window.localStorage.getItem("atlas-routing-mode")).toBe("research");
  });
});

describe("ChatPanel output cancellation", () => {
  it("shows a stop button while streaming and returns the composer to idle", () => {
    HTMLElement.prototype.scrollIntoView = vi.fn();
    useChatStore.setState({ streaming: true, isTyping: true, messages: [] });
    render(<ChatPanel activeDoc={null} onCitationClick={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Stop response" }));

    expect(useChatStore.getState().streaming).toBe(false);
    expect(useChatStore.getState().isTyping).toBe(false);
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
  });

  it("silently falls back to SSE when WebSocket authentication is stale", async () => {
    HTMLElement.prototype.scrollIntoView = vi.fn();
    useChatStore.setState({
      input: "Second question",
      streaming: false,
      isTyping: false,
      messages: [],
      activeSessionId: null,
    });
    vi.mocked(api.getValidAccessToken).mockResolvedValue("fresh-token");
    vi.mocked(api.streamPost).mockImplementation(async function* () {
      yield { type: "sources", data: [] };
      yield { type: "token", data: "Recovered answer" };
      yield { type: "done" };
    });

    class AuthFailingWebSocket {
      onopen: (() => void) | null = null;
      onmessage: ((event: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;

      constructor() {
        window.setTimeout(() => this.onopen?.(), 0);
      }

      send() {
        window.setTimeout(
          () =>
            this.onmessage?.({
              data: JSON.stringify({ type: "error", data: "Invalid or expired token" }),
            }),
          0,
        );
      }

      close() {
        window.setTimeout(() => this.onclose?.(), 0);
      }
    }
    vi.stubGlobal("WebSocket", AuthFailingWebSocket);

    render(<ChatPanel activeDoc={null} onCitationClick={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(screen.getByText("Recovered answer")).toBeInTheDocument());
    expect(screen.queryByText(/Invalid or expired token/i)).not.toBeInTheDocument();
  });
});
