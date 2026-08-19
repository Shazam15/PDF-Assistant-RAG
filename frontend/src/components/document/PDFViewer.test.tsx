import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PDFViewer from "./PDFViewer";

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Page: () => <div data-testid="pdf-page" />,
  pdfjs: { GlobalWorkerOptions: { workerSrc: "" } },
}));

describe("PDFViewer", () => {
  it("renders a close button that calls the close handler", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <PDFViewer
        documentId="doc-1"
        currentPage={1}
        onPageChange={() => undefined}
        totalPages={3}
        onClose={onClose}
      />
    );

    await user.click(screen.getByRole("button", { name: /close pdf viewer/i }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
