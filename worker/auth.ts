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
  emailVerified: boolean;
  name?: string;
  givenName?: string;
  familyName?: string;
  picture?: string;
  locale?: string;
  hostedDomain?: string;
  // every verified claim Google sent, stored as-is
  claims: Record<string, unknown>;
}

const str = (v: unknown): string | undefined => (typeof v === "string" && v ? v : undefined);

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
    emailVerified: payload.email_verified === true,
    name: str(payload.name),
    givenName: str(payload.given_name),
    familyName: str(payload.family_name),
    picture: str(payload.picture),
    locale: str(payload.locale),
    hostedDomain: str(payload.hd),
    claims: { ...payload },
  };
}

export const PHONE_SCOPE = "https://www.googleapis.com/auth/user.phonenumbers.read";

export interface GooglePhoneNumbers {
  primary: string | null;
  all: unknown[];
}

// Phone numbers aren't in the ID token. The browser gets a short-lived
// access token with the user.phonenumbers.read scope and hands it over once;
// we confirm with Google that it was issued to our client for this same
// user, read the numbers from the People API, and discard the token.
export async function fetchGooglePhoneNumbers(
  accessToken: string,
  clientId: string,
  expectedSub: string
): Promise<GooglePhoneNumbers> {
  const infoRes = await fetch(
    `https://oauth2.googleapis.com/tokeninfo?access_token=${encodeURIComponent(accessToken)}`
  );
  if (!infoRes.ok) throw new Error(`tokeninfo ${infoRes.status}`);
  const info = (await infoRes.json()) as { aud?: string; azp?: string; sub?: string; scope?: string };
  if ((info.aud ?? info.azp) !== clientId) throw new Error("access token was issued to another client");
  if (info.sub !== expectedSub) throw new Error("access token belongs to another google account");
  if (!info.scope?.split(" ").includes(PHONE_SCOPE)) throw new Error("phone number permission not granted");

  const res = await fetch("https://people.googleapis.com/v1/people/me?personFields=phoneNumbers", {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) throw new Error(`people api ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const person = (await res.json()) as {
    phoneNumbers?: { value?: string; canonicalForm?: string; metadata?: { primary?: boolean } }[];
  };
  const all = person.phoneNumbers ?? [];
  const primary = all.find((p) => p.metadata?.primary) ?? all[0];
  return { primary: primary ? primary.canonicalForm ?? primary.value ?? null : null, all };
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
