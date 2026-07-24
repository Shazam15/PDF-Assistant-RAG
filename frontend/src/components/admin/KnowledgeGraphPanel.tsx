"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Core, NodeSingular } from "cytoscape";
import {
  AlertTriangle,
  Expand,
  Network,
  RefreshCw,
  RotateCcw,
  Search,
} from "lucide-react";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

interface GraphSummary {
  document_id: string;
  document_name: string;
  owner_id: string;
  owner_username: string;
  node_count: number;
  edge_count: number;
}

interface GraphNode {
  id: string;
  name: string;
  label: string;
  mentions: number;
  degree: number;
  pages: number[];
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  pages: number[];
}

interface GraphDetail extends GraphSummary {
  nodes: GraphNode[];
  edges: GraphEdge[];
  returned_node_count: number;
  returned_edge_count: number;
  truncated: boolean;
}

const labelColors: Record<string, string> = {
  PERSON: "#16a34a",
  ORG: "#2563eb",
  GPE: "#dc2626",
  LOC: "#0891b2",
  PRODUCT: "#d97706",
  EVENT: "#9333ea",
  WORK_OF_ART: "#db2777",
  LAW: "#7c3aed",
  NORP: "#0f766e",
  FAC: "#4f46e5",
  UNKNOWN: "#64748b",
};

function pagesLabel(pages: number[]) {
  if (pages.length === 0) return "No page metadata";
  return `Pages ${pages.slice(0, 12).join(", ")}${pages.length > 12 ? "…" : ""}`;
}

function GraphCanvas({ graph }: { graph: GraphDetail }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [query, setQuery] = useState("");
  const [label, setLabel] = useState("all");

  const labels = useMemo(
    () => Array.from(new Set(graph.nodes.map((node) => node.label))).sort(),
    [graph.nodes],
  );

  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;
    let instance: Core | null = null;
    const visibleNodes = graph.nodes.filter((node) => label === "all" || node.label === label);
    const visibleIds = new Set(visibleNodes.map((node) => node.id));
    const visibleEdges = graph.edges.filter(
      (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
    );
    const maxWeight = Math.max(2, ...visibleEdges.map((edge) => edge.weight));

    void import("cytoscape").then(({ default: cytoscape }) => {
      if (cancelled || !containerRef.current) return;
      instance = cytoscape({
        container: containerRef.current,
        elements: [
          ...visibleNodes.map((node) => ({
            data: {
              ...node,
              color: labelColors[node.label] ?? labelColors.UNKNOWN,
            },
          })),
          ...visibleEdges.map((edge) => ({ data: edge })),
        ],
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(color)",
              "border-color": "#ffffff",
              "border-opacity": 0.75,
              "border-width": 1.5,
              height: "mapData(mentions, 1, 20, 24, 54)",
              width: "mapData(mentions, 1, 20, 24, 54)",
              label: "data(name)",
              color: "#111827",
              "font-size": 10,
              "font-weight": 600,
              "text-background-color": "#ffffff",
              "text-background-opacity": 0.88,
              "text-background-padding": "3px",
              "text-background-shape": "roundrectangle",
              "text-margin-y": -8,
              "text-max-width": "110px",
              "text-wrap": "ellipsis",
            },
          },
          {
            selector: "edge",
            style: {
              width: `mapData(weight, 1, ${maxWeight}, 1, 6)`,
              "line-color": "#64748b",
              opacity: 0.55,
              "curve-style": "haystack",
            },
          },
          {
            selector: "node:selected",
            style: {
              "border-color": "#f8fafc",
              "border-width": 4,
              "overlay-color": "#38bdf8",
              "overlay-opacity": 0.15,
            },
          },
          { selector: ".dimmed", style: { opacity: 0.12 } },
          {
            selector: ".matched",
            style: { "border-color": "#facc15", "border-width": 4, opacity: 1 },
          },
        ],
        layout: {
          name: "cose",
          animate: false,
          fit: true,
          padding: 40,
          nodeRepulsion: () => 7000,
          idealEdgeLength: () => 90,
        },
        minZoom: 0.15,
        maxZoom: 3,
      });
      cyRef.current = instance;
      instance.on("tap", "node", (event) => {
        setSelectedNode((event.target as NodeSingular).data() as GraphNode);
      });
      instance.on("tap", (event) => {
        if (event.target === instance) setSelectedNode(null);
      });
    });

    return () => {
      cancelled = true;
      instance?.destroy();
      if (cyRef.current === instance) cyRef.current = null;
    };
  }, [graph, label]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const normalized = query.trim().toLocaleLowerCase();
    cy.nodes().removeClass("dimmed matched");
    if (!normalized) return;
    const matches = cy.nodes().filter((node) =>
      String(node.data("name")).toLocaleLowerCase().includes(normalized),
    );
    cy.nodes().difference(matches).addClass("dimmed");
    matches.addClass("matched");
    if (matches.length > 0) cy.animate({ fit: { eles: matches, padding: 100 }, duration: 250 });
  }, [query, label]);

  const runLayout = () => {
    cyRef.current?.layout({ name: "cose", animate: true, animationDuration: 500 }).run();
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 lg:flex-row lg:items-center">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find an entity"
            className="pl-8"
          />
        </div>
        <select
          value={label}
          onChange={(event) => {
            setLabel(event.target.value);
            setSelectedNode(null);
          }}
          className="h-8 min-w-40 rounded-md border border-input bg-background px-2 text-sm"
          aria-label="Filter by entity type"
        >
          <option value="all">All entity types</option>
          {labels.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
        <div className="flex gap-1">
          <Button variant="outline" size="icon-sm" onClick={runLayout} title="Rearrange graph">
            <RotateCcw className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            onClick={() => cyRef.current?.fit(undefined, 40)}
            title="Fit graph"
          >
            <Expand className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid min-h-[540px] overflow-hidden rounded-md border border-border/70 xl:grid-cols-[minmax(0,1fr)_260px]">
        <div
          ref={containerRef}
          className="h-[540px] min-w-0 bg-background"
          aria-label={`Knowledge graph for ${graph.document_name}`}
        />
        <aside className="border-t border-border/70 bg-muted/20 p-4 xl:border-l xl:border-t-0">
          {selectedNode ? (
            <div className="space-y-4">
              <div className="space-y-1">
                <Badge
                  variant="outline"
                  style={{ borderColor: labelColors[selectedNode.label] ?? labelColors.UNKNOWN }}
                >
                  {selectedNode.label}
                </Badge>
                <h3 className="break-words text-base font-semibold">{selectedNode.name}</h3>
              </div>
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div><dt className="text-muted-foreground">Mentions</dt><dd>{selectedNode.mentions}</dd></div>
                <div><dt className="text-muted-foreground">Relations</dt><dd>{selectedNode.degree}</dd></div>
              </dl>
              <p className="text-sm text-muted-foreground">{pagesLabel(selectedNode.pages)}</p>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
              <Network className="h-6 w-6" />
              <p>Select an entity to inspect its evidence metadata.</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default function KnowledgeGraphPanel() {
  const [graphs, setGraphs] = useState<GraphSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [graph, setGraph] = useState<GraphDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadInventory = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get<{ items: GraphSummary[]; total: number }>("/api/v1/admin/graphs");
      setGraphs(response.items);
      setSelectedId((current) =>
        response.items.some((item) => item.document_id === current)
          ? current
          : response.items[0]?.document_id || "",
      );
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => void loadInventory(), 0);
    return () => window.clearTimeout(timeout);
  }, [loadInventory]);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    const timeout = window.setTimeout(() => {
      setLoading(true);
      setGraph(null);
      void api.get<GraphDetail>(`/api/v1/admin/graphs/${selectedId}?max_nodes=250`)
        .then((response) => {
          if (!cancelled) {
            setGraph(response);
            setError("");
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
        })
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [selectedId]);

  return (
    <Card>
      <CardHeader className="gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2 text-base">
            <Network className="h-4 w-4 text-primary" /> Knowledge graphs
          </CardTitle>
          <p className="mt-1 text-sm text-muted-foreground">
            Entity co-occurrences created during document ingestion.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void loadInventory()} disabled={loading}>
          <RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} /> Refresh
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && <div role="alert" className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</div>}
        {loading && graphs.length === 0 ? (
          <Skeleton className="h-[580px] w-full" />
        ) : graphs.length === 0 ? (
          <div className="flex min-h-48 flex-col items-center justify-center gap-2 border-y border-border/60 text-center text-sm text-muted-foreground">
            <Network className="h-7 w-7" />
            <p>No persisted knowledge graphs are available.</p>
          </div>
        ) : (
          <>
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <select
                value={selectedId}
                onChange={(event) => setSelectedId(event.target.value)}
                className="h-9 min-w-0 max-w-full rounded-md border border-input bg-background px-3 text-sm md:max-w-xl"
                aria-label="Select document graph"
              >
                {graphs.map((item) => (
                  <option key={item.document_id} value={item.document_id}>
                    {item.document_name} — {item.owner_username}
                  </option>
                ))}
              </select>
              {graph && (
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <span>{graph.node_count} entities</span><span>•</span><span>{graph.edge_count} relations</span>
                </div>
              )}
            </div>
            {graph?.truncated && (
              <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                Showing the 250 most connected entities from this graph.
              </div>
            )}
            {graph ? <GraphCanvas graph={graph} /> : <Skeleton className="h-[580px] w-full" />}
          </>
        )}
      </CardContent>
    </Card>
  );
}
