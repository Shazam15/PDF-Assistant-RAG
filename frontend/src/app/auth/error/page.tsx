"use client";

import { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { AlertTriangle, RefreshCcw, Home } from "lucide-react";

const ERROR_MESSAGES: Record<string, { title: string; description: string }> = {
  csrf_mismatch: {
    title: "Verificación de Seguridad Fallida",
    description:
      "El estado de OAuth no coincide. Esto podría indicar un ataque CSRF. Por favor, intenta iniciar sesión de nuevo.",
  },
  token_exchange_failed: {
    title: "Intercambio de Token Fallido",
    description:
      "No pudimos intercambiar tu código de autorización por un token de acceso. Por favor, intenta de nuevo.",
  },
  userinfo_failed: {
    title: "Recuperación de Perfil Fallida",
    description:
      "No pudimos recuperar tu perfil de Hugging Face. Por favor, verifica tu cuenta y vuelve a intentarlo.",
  },
  email_required: {
    title: "Email Requerido",
    description:
      "Tu cuenta de Hugging Face no proporcionó una dirección de email. Por favor, asegúrate de que tu email sea público y vuelve a intentarlo.",
  },
  oauth_not_configured: {
    title: "OAuth No ha sido configurado",
    description:
      "OAuth no está configurado en este servidor. Por favor, contacta al administrador.",
  },
  default: {
    title: "Autenticación Fallida",
    description:
      "Algo salió mal durante el inicio de sesión con Hugging Face. Por favor, intenta de nuevo.",
  },
};

function AuthErrorContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const errorCode = searchParams.get("error") ?? "default";
  const { title, description } =
    ERROR_MESSAGES[errorCode] ?? ERROR_MESSAGES.default;

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[500px] h-[300px] bg-destructive/8 rounded-full blur-[100px] pointer-events-none" />

      <Card className="w-full max-w-md relative z-10 bg-card/80 backdrop-blur-xl border-border/50">
        <CardHeader className="text-center pb-2">
          <div className="flex justify-center mb-4">
            <div className="w-12 h-12 rounded-xl bg-destructive/15 flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-destructive" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold">{title}</CardTitle>
          <CardDescription className="text-sm text-muted-foreground mt-1">
            {description}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3 pt-2">
          {errorCode !== "default" && (
            <div className="flex justify-center">
              <span className="text-xs font-mono bg-muted px-2 py-1 rounded-md text-muted-foreground">
                error: {errorCode}
              </span>
            </div>
          )}

          <Button
            className="w-full h-11"
            onClick={() => router.push("/login")}
          >
            <RefreshCcw className="w-4 h-4 mr-2" />
            Try Again
          </Button>

          <Button
            variant="outline"
            className="w-full h-11"
            onClick={() => router.push("/")}
          >
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default function AuthErrorPage() {
  return (
    <Suspense>
      <AuthErrorContent />
    </Suspense>
  );
}