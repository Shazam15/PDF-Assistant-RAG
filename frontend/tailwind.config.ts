import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";
import animate from "tailwindcss-animate";
import typography from "@tailwindcss/typography";

// Ported from the v4 `@import "shadcn/tailwind.css"` custom variants, which
// aren't available as `@custom-variant` CSS in Tailwind v3 — see
// frontend/README-avx2-fallback.md for why this file exists at all.
const dataStateVariants = plugin(({ addVariant }) => {
  addVariant("data-open", [
    '&[data-state="open"]',
    "&[data-open]:not([data-open='false'])",
  ]);
  addVariant("data-closed", [
    '&[data-state="closed"]',
    "&[data-closed]:not([data-closed='false'])",
  ]);
  addVariant("data-checked", [
    '&[data-state="checked"]',
    "&[data-checked]:not([data-checked='false'])",
  ]);
  addVariant("data-unchecked", [
    '&[data-state="unchecked"]',
    "&[data-unchecked]:not([data-unchecked='false'])",
  ]);
  addVariant("data-selected", '&[data-selected="true"]');
  addVariant("data-disabled", [
    '&[data-disabled="true"]',
    "&[data-disabled]:not([data-disabled='false'])",
  ]);
  addVariant("data-active", [
    '&[data-state="active"]',
    "&[data-active]:not([data-active='false'])",
  ]);
  addVariant("data-horizontal", '&[data-orientation="horizontal"]');
  addVariant("data-vertical", '&[data-orientation="vertical"]');
});

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        card: {
          DEFAULT: "var(--card)",
          foreground: "var(--card-foreground)",
        },
        popover: {
          DEFAULT: "var(--popover)",
          foreground: "var(--popover-foreground)",
        },
        primary: {
          DEFAULT: "var(--primary)",
          foreground: "var(--primary-foreground)",
        },
        secondary: {
          DEFAULT: "var(--secondary)",
          foreground: "var(--secondary-foreground)",
        },
        muted: {
          DEFAULT: "var(--muted)",
          foreground: "var(--muted-foreground)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          foreground: "var(--accent-foreground)",
        },
        destructive: "var(--destructive)",
        border: "var(--border)",
        input: "var(--input)",
        ring: "var(--ring)",
        chart: {
          "1": "var(--chart-1)",
          "2": "var(--chart-2)",
          "3": "var(--chart-3)",
          "4": "var(--chart-4)",
          "5": "var(--chart-5)",
        },
        sidebar: {
          DEFAULT: "var(--sidebar)",
          foreground: "var(--sidebar-foreground)",
          primary: "var(--sidebar-primary)",
          "primary-foreground": "var(--sidebar-primary-foreground)",
          accent: "var(--sidebar-accent)",
          "accent-foreground": "var(--sidebar-accent-foreground)",
          border: "var(--sidebar-border)",
          ring: "var(--sidebar-ring)",
        },
      },
      borderRadius: {
        sm: "calc(var(--radius) * 0.6)",
        md: "calc(var(--radius) * 0.8)",
        lg: "var(--radius)",
        xl: "calc(var(--radius) * 1.4)",
        "2xl": "calc(var(--radius) * 1.8)",
        "3xl": "calc(var(--radius) * 2.2)",
        "4xl": "calc(var(--radius) * 2.6)",
      },
      fontFamily: {
        sans: "var(--font-sans)",
        mono: "var(--font-geist-mono)",
        heading: "var(--font-sans)",
      },
    },
  },
  plugins: [animate, typography, dataStateVariants],
};

export default config;
