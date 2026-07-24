const protectedMarkdownCode = /(```[\s\S]*?```|`[^`\n]*`)/g;

function normalizeMathSegment(segment: string): string {
  return segment
    .replace(/\$[´`]([^$\n]+?)[´`]\$/g, (_match, formula: string) => `$${formula}$`)
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, formula: string) => `$$${formula}$$`)
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, formula: string) => `$${formula}$`);
}

/** Normalize common LLM delimiter variants without modifying Markdown code. */
export function normalizeMarkdownMath(markdown: string): string {
  return markdown
    .split(protectedMarkdownCode)
    .map((part, index) => (index % 2 === 0 ? normalizeMathSegment(part) : part))
    .join("");
}
