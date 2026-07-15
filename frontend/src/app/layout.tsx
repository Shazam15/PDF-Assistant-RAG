import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { TooltipProvider } from "@/components/ui/tooltip";
import I18nProvider from "@/components/providers/I18nProvider";
import { ThemeProvider } from "@/components/layout/ThemeProvider";
import { Toaster } from "sonner";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

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
    <html lang="en" className={`${inter.variable} h-full antialiased`} suppressHydrationWarning>
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
