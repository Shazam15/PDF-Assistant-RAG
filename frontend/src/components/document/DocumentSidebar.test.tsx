import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import DocumentSidebar from "./DocumentSidebar";
import type { DocInfo } from "@/app/dashboard/page";
import i18n from "@/lib/i18n";

beforeAll(async () => {
  await i18n.changeLanguage("en");
});

vi.mock("react-dropzone", () => ({
  useDropzone: () => ({
    getRootProps: () => ({}),
    getInputProps: () => ({}),
    isDragActive: false,
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    postForm: vi.fn(),
    delete: vi.fn(),
    renameDocument: vi.fn(),
  },
}));

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

const baseDocument: DocInfo = {
  id: "doc-1",
  original_name: "report.pdf",
  file_size: 1024,
  page_count: 12,
  chunk_count: 24,
  status: "processing",
  error_message: null,
  summary: "",
  uploaded_at: "2026-07-20T12:00:00Z",
};

function renderSidebar(document: DocInfo, onSelectDoc = vi.fn()) {
  render(
    <DocumentSidebar
      documents={[document]}
      activeDoc={null}
      onSelectDoc={onSelectDoc}
      onDocumentsChange={vi.fn()}
      onDocumentRenamed={vi.fn()}
    />,
  );
  return onSelectDoc;
}

describe("DocumentSidebar ingestion progress", () => {
  it("renders determinate embedding progress and batch counts", () => {
    renderSidebar({
      ...baseDocument,
      processing_progress: 60,
      processing_stage: "embedding",
      processing_current: 12,
      processing_total: 24,
    });

    expect(screen.getAllByText("Generating embeddings").length).toBeGreaterThan(0);
    expect(screen.getByText("12/24")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "60");
  });

  it("allows selecting a searchable document while enrichment continues", async () => {
    const onSelectDoc = renderSidebar({
      ...baseDocument,
      status: "enriching",
      processing_progress: 88,
      processing_stage: "summarizing",
      searchable_at: "2026-07-20T12:01:00Z",
    });

    expect(screen.getByText("Available")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Select document report.pdf. Status: enriching" }));
    expect(onSelectDoc).toHaveBeenCalledOnce();
  });
});
