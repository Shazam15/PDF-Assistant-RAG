import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "./api";

const makeJwt = (expiresAt: number) => {
  const encode = (value: object) =>
    window
      .btoa(JSON.stringify(value))
      .replace(/=/g, "")
      .replace(/\+/g, "-")
      .replace(/\//g, "_");
  return `${encode({ alg: "HS256", typ: "JWT" })}.${encode({ exp: expiresAt })}.signature`;
};

describe("ApiClient WebSocket authentication", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("reuses an access token that is not close to expiring", async () => {
    const token = makeJwt(Math.floor(Date.now() / 1000) + 600);
    window.localStorage.setItem("token", token);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const result = await new ApiClient("http://api.test").getValidAccessToken();

    expect(result).toBe(token);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refreshes an expired access token before opening a WebSocket", async () => {
    const expiredToken = makeJwt(Math.floor(Date.now() / 1000) - 10);
    const refreshedToken = makeJwt(Math.floor(Date.now() / 1000) + 900);
    window.localStorage.setItem("token", expiredToken);
    window.localStorage.setItem("refresh_token", "refresh-token");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          access_token: refreshedToken,
          refresh_token: "rotated-refresh-token",
          user: null,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await new ApiClient("http://api.test").getValidAccessToken();

    expect(result).toBe(refreshedToken);
    expect(window.localStorage.getItem("token")).toBe(refreshedToken);
    expect(window.localStorage.getItem("refresh_token")).toBe("rotated-refresh-token");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/v1/auth/refresh",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
