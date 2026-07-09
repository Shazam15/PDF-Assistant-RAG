"use client";

import { useId, useState } from "react";
import type { SourceBoundingBox, SourceChunk } from "@/store/chat-store";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ChevronDown, ChevronUp, FileText, Eye, TextQuote } from "lucide-react";

const EXCERPT_THRESHOLD = 200;

type ConfidenceLevel = "High" | "Medium" | "Low" | "Unknown";

interface ConfidenceBadgeMeta {
  label: ConfidenceLevel;
  className: string;
}

const normalizeMetricValue = (value?: number) => {
  if (typeof value !== "number" || Number.isNaN(value)) return undefined;
  return value > 1 ? value / 100 : value;
};

const formatMetricValue = (value?: number) => {
  const normalizedValue = normalizeMetricValue(value);
  if (normalizedValue === undefined) return "N/A";
  return `${Math.round(normalizedValue * 100)}%`;
};

const getConfidenceBadgeMeta = (value?: number): ConfidenceBadgeMeta => {
  const normalizedValue = normalizeMetricValue(value);

  if (normalizedValue === undefined) {
    return {
      label: "Desconocido",
      className: "border-muted bg-muted/40 text-muted-foreground",
    };
  }

  if (normalizedValue >= 0.8) {
    return {
      label: "Alta",
      className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-600",
    };
  }

  if (normalizedValue >= 0.5) {
    return {
      label: "Media",
      className: "border-amber-500/30 bg-amber-500/10 text-amber-600",
    };
  }

  return {
    label: "Baja",
    className: "border-red-500/30 bg-red-500/10 text-red-600",
  };
};

const getPrimarySourceMetric = (source: SourceChunk) =>
  source.confidence ?? source.score;

const MetricBadge = ({
  label,
  value,
}: {
  label: "Score" | "Confianza";
  value?: number;
}) => {
  const badgeMeta = getConfidenceBadgeMeta(value);

  return (
    <Badge
      variant="outline"
      className={`h-5 px-1.5 text-[9px] font-medium ${badgeMeta.className}`}
      title={`${label}: ${formatMetricValue(value)}`}
    >
      {label}: {badgeMeta.label}
    </Badge>
  );
};

interface Props {
  sources: SourceChunk[];
  onPageClick: (payload: {
    page: number;
    highlightRects?: SourceBoundingBox[];
  }) => void;
}

export default function SourceCard({ sources = [], onPageClick }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [excerptOpen, setExcerptOpen] = useState<Set<number>>(new Set());
  const sourceListId = useId();

  if (sources.length === 0) return null;

  const toggleExcerpt = (index: number) => {
    const next = new Set(excerptOpen);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    setExcerptOpen(next);
  };

  return (
    <div className="rounded-lg border border-border/50 bg-card/50 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls={sourceListId}
        aria-label={`${expanded ? "Collapse" : "Expand"} ${sources.length} cited source${sources.length > 1 ? "s" : ""}`}
        className="w-full flex items-center justify-between px-3 py-2 text-xs hover:bg-accent/30 transition-colors"
      >
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <FileText className="w-3.5 h-3.5" />
          {sources.length} source{sources.length > 1 ? "s" : ""} cited
        </span>
        {expanded ? (
          <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
        )}
      </button>

      {!expanded && (
        <div className="px-3 pb-2 flex flex-wrap gap-1">
          {sources.map((src, i) => {
            const badgeMeta = getConfidenceBadgeMeta(
              getPrimarySourceMetric(src)
            );
            const isWebSource = src.source_type === "web";
            const label = isWebSource ? src.source_id || `W${i + 1}` : `p.${src.page}`;

            return (
              <Tooltip key={i}>
                <TooltipTrigger
                  type="button"
                  className="inline-flex"
                  onClick={() => {
                    if (!isWebSource) {
                      onPageClick({
                        page: src.page,
                        highlightRects: src.highlightRects,
                      });
                    }
                  }}
                  aria-label={
                    isWebSource
                      ? `Open web source ${label}. Confidence ${badgeMeta.label}`
                      : `Go to source page ${src.page}. Confidence ${badgeMeta.label}`
                  }
                >
                  <Badge
                    variant="outline"
                    className={`text-[10px] h-5 cursor-pointer hover:bg-primary/20 transition-colors ${badgeMeta.className}`}
                  >
                    {label} - {badgeMeta.label}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent
                  side="top"
                  align="center"
                  className="max-w-xs p-2"
                >
                  <div className="mb-1 flex flex-wrap gap-1">
                    <MetricBadge label="Score" value={src.score} />
                    <MetricBadge label="Confianza" value={src.confidence} />
                  </div>
                  <p className="text-[11px] leading-relaxed line-clamp-6">
                    {src.source_type === "web" ? src.snippet || src.text : src.text}
                  </p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      )}

      {expanded && (
        <div id={sourceListId} className="border-t border-border/30">
          {sources.map((src, i) => (
            <div
              key={i}
              className="px-3 py-2.5 border-b border-border/20 last:border-b-0 hover:bg-accent/20 transition-colors"
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="truncate text-[10px] font-medium text-muted-foreground">
                    {src.source_type === "web" ? src.title || src.filename : src.filename}
                  </span>
                  {src.source_type === "web" ? (
                    <Badge variant="outline" className="h-5 px-1.5 text-[9px]">
                      {src.source_id || `W${i + 1}`}
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="h-5 px-1.5 text-[9px]">
                      Page {src.page}
                    </Badge>
                  )}
                  <MetricBadge label="Score" value={src.score} />
                  <MetricBadge label="Confianza" value={src.confidence} />
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 shrink-0 px-2 text-[10px]"
                  onClick={() => {
                    if (src.source_type === "web" && src.url) {
                      window.open(src.url, "_blank", "noopener,noreferrer");
                      return;
                    }
                    onPageClick({
                      page: src.page,
                      highlightRects: src.highlightRects,
                    });
                  }}
                  aria-label={
                    src.source_type === "web"
                      ? `View web source ${src.source_id || i + 1}`
                      : `View source page ${src.page}`
                  }
                >
                  <Eye className="w-3 h-3 mr-1" />
                  View
                </Button>
              </div>
              <p
                className={`text-[11px] text-muted-foreground leading-relaxed ${
                  excerptOpen.has(i) ? "" : "line-clamp-3"
                }`}
              >
                {src.source_type === "web" ? src.snippet || src.text : src.text}
              </p>
              {src.text.length > EXCERPT_THRESHOLD && (
                <button
                  type="button"
                  onClick={() => toggleExcerpt(i)}
                  aria-expanded={excerptOpen.has(i)}
                  className="mt-1.5 flex items-center gap-1 text-[10px] text-primary/70 hover:text-primary transition-colors"
                >
                  <TextQuote className="w-3 h-3" />
                  {excerptOpen.has(i) ? "Hide excerpt" : "Show excerpt"}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
