import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { TooltipProvider } from "@/components/ui/tooltip";
import I18nProvider from "@/components/providers/I18nProvider";
import { ThemeProvider } from "@/components/layout/ThemeProvider";
import { Toaster } from "sonner";

// `next/font` requires SWC and is incompatible with the Babel fallback this
// AVX2-compatible branch uses (see frontend/README-avx2-fallback.md), so
// Inter is loaded via a plain Google Fonts stylesheet in <head> below
// instead, with --font-sans set in globals.css.

export const metadata: Metadata = {
  title: "A.T.L.A.S. - AI Text and Language Analysis System",
  description:
    "Suba archivos PDF complejos y chatee con un agente de IA que extrae información específica, resume datos y cita fuentes con precisión utilizando Generación Aumentada por Recuperación.",
  keywords: ["RAG", "Document AI", "Análisis de PDF", "LLM", "Vector Search"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
          themes={["light", "dark", "ocean", "forest", "sunset"]}
        >
          <AuthProvider>
            <I18nProvider>
              <TooltipProvider>
                {children}
              </TooltipProvider>
            </I18nProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
