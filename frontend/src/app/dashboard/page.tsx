"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { toast } from "sonner";
import { api, CONNECTION_ERROR_BANNER_MESSAGE, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import Header from "@/components/layout/Header";
import DocumentSidebar from "@/components/document/DocumentSidebar";
import ChatSessionSidebar from "@/components/chat/ChatSessionSidebar";
import ChatPanel from "@/components/chat/ChatPanel";
function PDFViewerSkeleton() {
  return (
    <div
      className="h-full flex flex-col bg-background"
      aria-busy="true"
      aria-label="Loading PDF viewer"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 bg-card/50 shrink-0">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-md bg-muted/70 animate-pulse" />
          <div className="h-7 w-20 rounded-md bg-muted/70 animate-pulse" />
          <div className="h-7 w-7 rounded-md bg-muted/70 animate-pulse" />
        </div>
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-md bg-muted/70 animate-pulse" />
          <div className="h-4 w-10 rounded bg-muted/70 animate-pulse" />
          <div className="h-7 w-7 rounded-md bg-muted/70 animate-pulse" />
        </div>
      </div>
      <div className="flex-1 p-4">
        <div className="h-full rounded-lg border border-border/50 bg-muted/40 animate-pulse" />
      </div>
    </div>
  );
}

const PDFViewer = dynamic(() => import("@/components/document/PDFViewer"), {
  ssr: false,
  loading: () => <PDFViewerSkeleton />,
});

export interface DocInfo {
  chunk_size?: number;
  chunk_overlap?: number;
  summary: string;
  id: string;
  original_name: string;
  file_size: number;
  page_count: number;
  chunk_count: number;
  status: string;
  error_message: string | null;
  uploaded_at: string;
}

interface DocumentListPayload {
  documents?: DocInfo[];
  items?: DocInfo[];
  total?: number;
  page?: number;
  pages?: number;
}

export default function DashboardPage() {
  const { user, loading, initialized } = useAuth();
  const router = useRouter();

  const [documents, setDocuments] = useState<DocInfo[]>([]);
  const prevDocsRef = useRef<Record<string, string>>({});
  const [activeDoc, setActiveDoc] = useState<DocInfo | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfHighlightTarget, setPdfHighlightTarget] = useState<{
    page: number;
    rects?: {
      left: number;
      top: number;
      width: number;
      height: number;
      unit?: "percent" | "pixels" | "pdf";
    }[];
  } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [viewerOpen, setViewerOpen] = useState(true);
  const [connectionError, setConnectionError] = useState("");
  const [documentsLoading, setDocumentsLoading] = useState(true);

  const resetViewerState = useCallback(() => {
    setViewerOpen(false);
    setActiveDoc(null);
    setPdfPage(1);
    setPdfHighlightTarget(null);
  }, []);

  const handleSelectDoc = useCallback((doc: DocInfo) => {
    setActiveDoc(doc);
    setPdfPage(1);
    setPdfHighlightTarget(null);
    if (doc.original_name.toLowerCase().endsWith(".pdf")) {
      setViewerOpen(true);
    } else {
      setViewerOpen(false);
    }
  }, []);

  const handleDocumentRenamed = useCallback((renamedDocument: DocInfo) => {
    setDocuments((current) =>
      current.map((document) => (document.id === renamedDocument.id ? renamedDocument : document))
    );
    setActiveDoc((current) => (current?.id === renamedDocument.id ? renamedDocument : current));
  }, []);

  const documentsEqual = useCallback((a: DocInfo[], b: DocInfo[]) => {
    if (a.length !== b.length) return false;

    for (let i = 0; i < a.length; i += 1) {
      const lhs = a[i];
      const rhs = b[i];

      if (
        lhs.id !== rhs.id ||
        lhs.original_name !== rhs.original_name ||
        lhs.file_size !== rhs.file_size ||
        lhs.page_count !== rhs.page_count ||
        lhs.chunk_count !== rhs.chunk_count ||
        lhs.status !== rhs.status ||
        lhs.error_message !== rhs.error_message ||
        lhs.uploaded_at !== rhs.uploaded_at ||
        lhs.summary !== rhs.summary ||
        lhs.chunk_size !== rhs.chunk_size ||
        lhs.chunk_overlap !== rhs.chunk_overlap
      ) {
        return false;
      }
    }

    return true;
  }, []);

  // Auth guard

  useEffect(() => {
    if (initialized && !user) router.replace("/login");
  }, [user, initialized, router]);

  // Check if Hugging Face token configuration is present
  useEffect(() => {
    if (user) {
      const hasHfToken = !!(user.hf_token || localStorage.getItem("hf_token"));

      if (!hasHfToken) {
        console.info(
          "No hay token de API de Hugging Face configurado. El acceso al modelo personal volverá al valor predeterminado del sistema a menos que se establezca en el menú del perfil del usuario."
        );
      }
    }
  }, [user]);


  // Load documents
  const loadDocuments = useCallback(async (skipLoading = false) => {
    if (!skipLoading) {
      setDocumentsLoading(true);
    }

    try {
      const firstPage = await api.get<DocumentListPayload>("/api/v1/documents/");
      const firstPageDocuments = firstPage?.documents ?? firstPage?.items ?? [];
      let nextDocuments = firstPageDocuments;

      if (firstPage?.items && (firstPage.pages ?? 1) > 1) {
        const remainingPages = Array.from(
          { length: (firstPage.pages ?? 1) - 1 },
          (_, index) => index + 2
        );

        const remainingResponses = await Promise.all(
          remainingPages.map((page) =>
            api.get<DocumentListPayload>(`/api/v1/documents/?page=${page}`)
          )
        );

        nextDocuments = [
          ...firstPageDocuments,
          ...remainingResponses.flatMap(
            (response) => response?.items ?? response?.documents ?? []
          ),
        ];
      }

      setDocuments((current) =>
        documentsEqual(current, nextDocuments) ? current : nextDocuments
      );
      setConnectionError("");
    } catch (err) {
      const message = err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE;
      setConnectionError(
        message === CONNECTION_ERROR_MESSAGE
          ? CONNECTION_ERROR_BANNER_MESSAGE
          : `⚠️ ${message}`
      );
    } finally {
      if (!skipLoading) {
        setDocumentsLoading(false);
      }
    }
  }, [documentsEqual]);

  useEffect(() => {
    if (!user) return;
    void (async () => {
      await loadDocuments();
    })();
  }, [user, loadDocuments]);

  // Ingest status change toast notification handler
  useEffect(() => {
    const prev = prevDocsRef.current;
    const nextPrevDocs: Record<string, string> = {};
    (documents || []).forEach((doc) => {
      nextPrevDocs[doc.id] = doc.status;

      const oldStatus = prev[doc.id];
      if (oldStatus && oldStatus !== doc.status) {
        if (doc.status === "ready") {
          toast.success(`🎉 Ingestion completa: '${doc.original_name}' está listo!`);
        } else if (doc.status === "failed") {
          toast.error(`❌ Ingestion fallida para '${doc.original_name}': ${doc.error_message || "Error desconocido"}`);
        }
      }
    });
    prevDocsRef.current = nextPrevDocs;
  }, [documents]);

  // Poll for processing status
  useEffect(() => {
    const hasPending = (documents || []).some(
      (d) => d.status === "pending" || d.status === "processing"
    );
    if (!hasPending) return;

    const interval = setInterval(() => void loadDocuments(true), 3000);
    return () => clearInterval(interval);
  }, [documents, loadDocuments]);

  if (!initialized || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse-glow w-12 h-12 rounded-full bg-primary/20" />
      </div>
    );
  }

  // Shared sidebar content — used by both desktop panel and mobile sheet
  const sidebarContent = (
    <DocumentSidebar
      documents={documents}
      activeDoc={activeDoc}
      loading={documentsLoading}
      onSelectDoc={handleSelectDoc}
      onDocumentsChange={loadDocuments}
      onDocumentRenamed={handleDocumentRenamed}
    />
  );

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        viewerOpen={viewerOpen}
        onToggleViewer={() => {
          if (viewerOpen) {
            resetViewerState();
          } else if (activeDoc?.original_name.toLowerCase().endsWith(".pdf")) {
            setViewerOpen(true);
          }
        }}
        mobileSheetContent={sidebarContent}
      />

      {connectionError && (
        <div
          role="alert"
          className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive"
        >
          {connectionError}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* ── Left: Document Sidebar — desktop only (md+) ─────────── */}
        {sidebarOpen && (
          <div className="hidden md:block w-72 flex-shrink-0 border-r border-border/50 overflow-hidden animate-fade-in-up">
            {sidebarContent}
          </div>
        )}

        {/* ── Left-Center: Chat Sessions Sidebar ──── */}
        <ChatSessionSidebar />

        {/* ── Center: Chat Panel ──────────────────────────────────── */}
        <div className="flex-1 min-w-0 flex flex-col">
          <ChatPanel
            activeDoc={activeDoc}
            onCitationClick={(target) => {
              setPdfPage(target.page);
              setPdfHighlightTarget({ page: target.page, rects: target.highlightRects });
              if (!viewerOpen) setViewerOpen(true);
            }}
          />
        </div>

        {/* ── Right: PDF Viewer — hidden on mobile ────────────────── */}
        {viewerOpen && activeDoc && activeDoc.original_name.endsWith(".pdf") && (
          <div className="hidden md:block w-[480px] flex-shrink-0 border-l border-border/50 overflow-hidden animate-fade-in-up">
            <PDFViewer
              documentId={activeDoc.id}
              currentPage={pdfPage}
              onPageChange={(page) => {
                setPdfPage(page);
                if (pdfHighlightTarget?.page !== page) {
                  setPdfHighlightTarget(null);
                }
              }}
              totalPages={activeDoc.page_count}
              highlightTarget={pdfHighlightTarget}
              onClose={resetViewerState}
            />
          </div>
        )}
      </div>
    </div>
  );
}
