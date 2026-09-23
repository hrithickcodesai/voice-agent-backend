import { createRemoteJWKSet, jwtVerify, SignJWT } from "jose";

// Sign in with Google: the browser gets an ID token from Google Identity
// Services, we verify it here against Google's published keys, then issue
// our own session cookie (an HS256 JWT signed with SESSION_SECRET).

const GOOGLE_JWKS = createRemoteJWKSet(new URL("https://www.googleapis.com/oauth2/v3/certs"));
const GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"];

export const SESSION_COOKIE = "cf_user_session";
const SESSION_DAYS = 30;

export interface GoogleProfile {
  sub: string;
  email: string;
  name?: string;
  picture?: string;
}

export async function verifyGoogleCredential(credential: string, clientId: string): Promise<GoogleProfile> {
  const { payload } = await jwtVerify(credential, GOOGLE_JWKS, {
    issuer: GOOGLE_ISSUERS,
    audience: clientId,
  });
  if (!payload.sub || typeof payload.email !== "string" || payload.email_verified !== true) {
    throw new Error("google account email missing or unverified");
  }
  return {
    sub: payload.sub,
    email: payload.email,
    name: typeof payload.name === "string" ? payload.name : undefined,
    picture: typeof payload.picture === "string" ? payload.picture : undefined,
  };
}

function key(secret: string): Uint8Array {
  return new TextEncoder().encode(secret);
}

export async function createSessionToken(userId: string, secret: string): Promise<string> {
  return new SignJWT({})
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(userId)
    .setIssuedAt()
    .setExpirationTime(`${SESSION_DAYS}d`)
    .sign(key(secret));
}

export function readCookie(request: Request, name: string): string | null {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name && rest.length) return rest.join("=");
  }
  return null;
}

// The signed-in user's id, or null (no cookie, bad signature, expired).
export async function sessionUserId(request: Request, secret: string | undefined): Promise<string | null> {
  const token = readCookie(request, SESSION_COOKIE);
  if (!token || !secret) return null;
  try {
    const { payload } = await jwtVerify(token, key(secret), { algorithms: ["HS256"] });
    return payload.sub ?? null;
  } catch {
    return null;
  }
}

export function sessionCookie(token: string): string {
  return `${SESSION_COOKIE}=${token}; Path=/; Max-Age=${SESSION_DAYS * 86400}; HttpOnly; Secure; SameSite=Lax`;
}

export function clearedSessionCookie(): string {
  return `${SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax`;
}
