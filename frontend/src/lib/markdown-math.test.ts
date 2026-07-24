import { describe, expect, it } from "vitest";
import { normalizeMarkdownMath } from "./markdown-math";

describe("normalizeMarkdownMath", () => {
  it("repairs accented and bracketed math delimiters", () => {
    expect(normalizeMarkdownMath(String.raw`$´\sqrt{x}´$`)).toBe(
      String.raw`$\sqrt{x}$`,
    );
    expect(normalizeMarkdownMath(String.raw`\[x^2 + 1\]`)).toBe(
      String.raw`$$x^2 + 1$$`,
    );
    expect(normalizeMarkdownMath(String.raw`\(x + 1\)`)).toBe(
      String.raw`$x + 1$`,
    );
  });

  it("does not rewrite inline or fenced code", () => {
    const markdown = "Code: `$´x´$`\n\n```tex\n$´x´$\n```";

    expect(normalizeMarkdownMath(markdown)).toBe(markdown);
  });
});
