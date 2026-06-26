"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import {
  FileText,
  MessageSquare,
  Brain,
  Shield,
  Zap,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import ContributorsPanel from "@/components/layout/ContributorsPanel";
import OpenSourceBadge from "@/components/layout/OpenSourceBadge";

// ─── Feature card data ───────────────────────────────────────────────────────
const FEATURES = [
  {
    icon: FileText,
    title: "Carga de múltiples formatos",
    desc: "PDF, DOCX, TXT, y Markdown usando chunking inteligente",
  },
  {
    icon: Search,
    title: "Búsqueda Semántica",
    desc: "Recuperación en dos etapas con reordenación mediante codificadores cruzados",
  },
  {
    icon: MessageSquare,
    title: "Chat en vivo",
    desc: "Respuestas en tiempo real de IA con citas de fuentes",
  },
  {
    icon: Zap,
    title: "Información al Instante",
    desc: "Extraer datos clave, resúmenes y puntos de información",
  },
  {
    icon: Shield,
    title: "Aislamiento de Datos",
    desc: "Colecciones vectoriales por usuario para privacidad completa",
  },
  {
    icon: Brain,
    title: "LLMs de Código Abierto",
    desc: "Con tecnología de Mistral y el ecosistema de HuggingFace",
  },
];

// ─── Page ────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [hallOfFameOpen, setHallOfFameOpen] = useState(false);

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse w-12 h-12 rounded-full bg-primary/20" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="relative flex-1 flex flex-col items-center justify-center overflow-hidden px-6 py-28">

        {/* Ambient glow — top-center */}
        <div
          aria-hidden
          className="pointer-events-none absolute top-0 left-1/2 -translate-x-1/2
                     w-[700px] h-[420px]
                     bg-primary/10 rounded-full blur-[120px]"
        />

        {/* Subtle secondary glow — bottom-right */}
        <div
          aria-hidden
          className="pointer-events-none absolute bottom-0 right-0
                     w-[400px] h-[300px]
                     bg-primary/5 rounded-full blur-[100px]"
        />

        {/* Fine grid overlay */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(hsl(var(--foreground)) 1px, transparent 1px)," +
              "linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        {/* ── Content ─────────────────────────────────────────────────────── */}
        <div className="relative z-10 text-center max-w-3xl mx-auto space-y-8 animate-fade-in-up">

          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                          bg-primary/10 border border-primary/20
                          text-xs font-medium tracking-wide text-primary uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 shadow-[0_0_6px_theme(colors.green.400)] animate-pulse" />
            Agentic Text Learning & Analysis System
          </div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.08]">
            A.T.L.A.S
          </h1>

          {/* Divider glow line */}
          <div className="mx-auto w-24 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

          {/* Sub-copy */}
          <p className="text-base sm:text-lg text-muted-foreground max-w-xl mx-auto leading-relaxed font-light">
            Transforma tus documentos, papers académicos y literatura científica en un entorno de conocimiento 
            interactivo. Consulta a tu agente, descubre conexiones ocultas y obtén respuestas precisas con citas 
            directas a las fuentes
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap gap-3 justify-center pt-2">
            <Link href="/register">
              <Button
                size="lg"
                className="px-8 text-base h-12 shadow-lg shadow-primary/20
                           hover:shadow-primary/30 hover:-translate-y-0.5
                           transition-all duration-200"
              >
                Registrarse
              </Button>
            </Link>

            <Link href="/login">
              <Button
                size="lg"
                variant="outline"
                className="px-8 text-base h-12
                           hover:-translate-y-0.5 transition-all duration-200"
              >
                Ingresar
              </Button>
            </Link>

          </div>
        </div>

        {/* ── Feature cards ────────────────────────────────────────────────── */}
        <div className="relative z-10 mt-20 w-full max-w-4xl mx-auto
                        grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FEATURES.map(({ icon: Icon, title, desc }, i) => (
            <div
              key={title}
              className="group p-5 rounded-xl
                         border border-border/40 bg-card/40 backdrop-blur-sm
                         hover:border-primary/25 hover:bg-card/70
                         transition-all duration-300 ease-out
                         hover:-translate-y-0.5"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              {/* Icon with subtle glow on hover */}
              <div className="w-8 h-8 mb-4 rounded-lg bg-primary/10 flex items-center justify-center
                              group-hover:bg-primary/15 transition-colors duration-300">
                <Icon className="w-4 h-4 text-primary group-hover:scale-110 transition-transform duration-300" />
              </div>

              <h3 className="font-semibold text-sm mb-1.5 text-foreground">{title}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="py-6 border-t border-border/40">
        <p className="text-center text-xs text-muted-foreground/60 tracking-wide">
          Creado con{" "}
          {["FastAPI", "LangChain", "ChromaDB", "HuggingFace", "Next.js", "Ollama"].map(
            (tech, i, arr) => (
              <span key={tech}>
                <span className="text-muted-foreground">{tech}</span>
                {i < arr.length - 1 && (
                  <span className="mx-1.5 text-muted-foreground/30">·</span>
                )}
              </span>
            )
          )}
        </p>
      </footer>

      {/* Hall of Fame modal */}
      {hallOfFameOpen && (
        <ContributorsPanel onClose={() => setHallOfFameOpen(false)} />
      )}

      {/* Floating open-source badge */}
      <OpenSourceBadge onOpenHallOfFame={() => setHallOfFameOpen(true)} />
    </div>
  );
}
