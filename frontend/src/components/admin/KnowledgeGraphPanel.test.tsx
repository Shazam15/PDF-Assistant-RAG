import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import KnowledgeGraphPanel from "./KnowledgeGraphPanel";
import { api } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: { get: vi.fn() },
  CONNECTION_ERROR_MESSAGE: "Connection failed",
}));

vi.mock("cytoscape", () => ({
  default: vi.fn(() => {
    const collection = {
      length: 0,
      removeClass: vi.fn().mockReturnThis(),
      addClass: vi.fn().mockReturnThis(),
      difference: vi.fn().mockReturnThis(),
      filter: vi.fn().mockReturnThis(),
    };
    return {
      nodes: vi.fn(() => collection),
      on: vi.fn(),
      destroy: vi.fn(),
      fit: vi.fn(),
      animate: vi.fn(),
      layout: vi.fn(() => ({ run: vi.fn() })),
    };
  }),
}));

describe("KnowledgeGraphPanel", () => {
  it("loads the admin graph inventory and selected graph", async () => {
    vi.mocked(api.get).mockImplementation(async (path: string) => {
      if (path === "/api/v1/admin/graphs") {
        return {
          total: 1,
          items: [{
            document_id: "doc-1",
            document_name: "engine-study.pdf",
            owner_id: "user-1",
            owner_username: "researcher",
            node_count: 2,
            edge_count: 1,
          }],
        } as never;
      }
      return {
        document_id: "doc-1",
        document_name: "engine-study.pdf",
        owner_id: "user-1",
        owner_username: "researcher",
        node_count: 2,
        edge_count: 1,
        returned_node_count: 2,
        returned_edge_count: 1,
        truncated: false,
        nodes: [
          { id: "ORG:a", name: "A", label: "ORG", mentions: 2, degree: 1, pages: [1] },
          { id: "PRODUCT:b", name: "B", label: "PRODUCT", mentions: 1, degree: 1, pages: [2] },
        ],
        edges: [{ id: "a::b", source: "ORG:a", target: "PRODUCT:b", weight: 1, pages: [2] }],
      } as never;
    });

    render(<KnowledgeGraphPanel />);

    expect(await screen.findByRole("option", { name: "engine-study.pdf — researcher" })).toBeInTheDocument();
    expect(await screen.findByText("2 entities")).toBeInTheDocument();
    expect(screen.getByText("1 relations")).toBeInTheDocument();
    await waitFor(() => {
      expect(api.get).toHaveBeenCalledWith("/api/v1/admin/graphs/doc-1?max_nodes=250");
    });
  });
});
