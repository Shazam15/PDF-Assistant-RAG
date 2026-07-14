import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageBubble from "./MessageBubble";
import ChatPanel from "./ChatPanel";
import { useChatStore, type ChatMsg } from "@/store/chat-store";

vi.mock("@/lib/api", () => ({
  api: {
    patch: vi.fn(),
    post: vi.fn(),
  },
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
});
