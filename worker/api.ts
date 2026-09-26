import {
  clearedSessionCookie,
  createSessionToken,
  fetchGooglePhoneNumbers,
  PHONE_SCOPE,
  sessionCookie,
  sessionUserId,
  verifyGoogleCredential,
} from "./auth";

export interface ApiEnv {
  DB: D1Database;
  GOOGLE_CLIENT_ID: string;
  SESSION_SECRET: string;
  MAX_CALL_MINUTES?: string;
}

// Sign-in is only enforced once a Google client id is configured, so the
// app keeps working (anonymously, without a call log) until then.
export function authEnabled(env: ApiEnv): boolean {
  return Boolean(env.GOOGLE_CLIENT_ID && env.SESSION_SECRET);
}

const MAX_MESSAGES = 1000;
const MAX_MESSAGE_CHARS = 4000;

function json(status: number, body: unknown, headers: HeadersInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers },
  });
}

async function readJson<T>(request: Request): Promise<T | null> {
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

interface UserRow {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  phone_number: string | null;
  phone_prompted_at: number | null;
}

// what the browser gets about the signed-in user
const USER_COLUMNS = "id, email, name, picture, phone_number, phone_prompted_at";
function publicUser(u: UserRow) {
  return {
    id: u.id,
    email: u.email,
    name: u.name,
    picture: u.picture,
    phone_number: u.phone_number,
    // offer the one-time "add your phone number" step after signup
    needs_phone_prompt: !u.phone_number && !u.phone_prompted_at,
  };
}

async function signIn(request: Request, env: ApiEnv): Promise<Response> {
  const body = await readJson<{ credential?: string }>(request);
  if (!body?.credential) return json(400, { error: "credential required" });

  let profile;
  try {
    profile = await verifyGoogleCredential(body.credential, env.GOOGLE_CLIENT_ID);
  } catch (err) {
    console.warn("google sign-in rejected", err);
    return json(401, { error: "invalid google credential" });
  }

  // Everything Google shared is kept: profile columns refresh on each
  // sign-in, signup_claims is written once and never overwritten, and
  // google_claims always holds the latest full claim set.
  const now = Date.now();
  const claims = JSON.stringify(profile.claims);
  const user = await env.DB.prepare(
    `INSERT INTO users (id, google_sub, email, name, picture, created_at, last_login_at,
                        given_name, family_name, email_verified, locale, hosted_domain,
                        signup_claims, google_claims)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
     ON CONFLICT (google_sub) DO UPDATE SET
       email = excluded.email, name = excluded.name, picture = excluded.picture,
       given_name = excluded.given_name, family_name = excluded.family_name,
       email_verified = excluded.email_verified, locale = excluded.locale,
       hosted_domain = excluded.hosted_domain, google_claims = excluded.google_claims,
       last_login_at = excluded.last_login_at
     RETURNING ${USER_COLUMNS}`
  )
    .bind(
      crypto.randomUUID(),
      profile.sub,
      profile.email,
      profile.name ?? null,
      profile.picture ?? null,
      now,
      profile.givenName ?? null,
      profile.familyName ?? null,
      profile.emailVerified ? 1 : 0,
      profile.locale ?? null,
      profile.hostedDomain ?? null,
      claims
    )
    .first<UserRow>();

  const token = await createSessionToken(user!.id, env.SESSION_SECRET);
  return json(200, { user: publicUser(user!) }, { "Set-Cookie": sessionCookie(token) });
}

async function savePhoneFromGoogle(request: Request, env: ApiEnv, userId: string): Promise<Response> {
  const body = await readJson<{ access_token?: string }>(request);
  if (!body?.access_token) return json(400, { error: "access_token required" });
  const row = await env.DB.prepare(`SELECT google_sub FROM users WHERE id = ?1`)
    .bind(userId)
    .first<{ google_sub: string }>();
  if (!row) return json(401, { error: "sign in required" });

  let phones;
  try {
    phones = await fetchGooglePhoneNumbers(body.access_token, env.GOOGLE_CLIENT_ID, row.google_sub);
  } catch (err) {
    console.warn("google phone lookup failed", err);
    return json(400, { error: String(err instanceof Error ? err.message : err) });
  }
  const user = await env.DB.prepare(
    `UPDATE users SET phone_number = ?1, phone_numbers = ?2, phone_prompted_at = ?3
     WHERE id = ?4 RETURNING ${USER_COLUMNS}`
  )
    .bind(phones.primary, JSON.stringify(phones.all), Date.now(), userId)
    .first<UserRow>();
  return json(200, { user: publicUser(user!), found: phones.all.length });
}

async function skipPhone(env: ApiEnv, userId: string): Promise<Response> {
  const user = await env.DB.prepare(
    `UPDATE users SET phone_prompted_at = ?1 WHERE id = ?2 RETURNING ${USER_COLUMNS}`
  )
    .bind(Date.now(), userId)
    .first<UserRow>();
  return json(200, { user: publicUser(user!) });
}

async function listCalls(env: ApiEnv, userId: string): Promise<Response> {
  const { results } = await env.DB.prepare(
    `SELECT id, kind, status, started_at, ended_at, duration_s
     FROM calls WHERE user_id = ?1 ORDER BY started_at DESC LIMIT 200`
  )
    .bind(userId)
    .all();
  return json(200, { calls: results });
}

async function createCall(request: Request, env: ApiEnv, userId: string): Promise<Response> {
  const body = await readJson<{ kind?: string }>(request);
  const kind = body?.kind === "facetime" ? "facetime" : "audio";
  const id = crypto.randomUUID();
  await env.DB.prepare(
    `INSERT INTO calls (id, user_id, kind, status, started_at, updated_at) VALUES (?1, ?2, ?3, 'active', ?4, ?4)`
  )
    .bind(id, userId, kind, Date.now())
    .run();
  return json(201, { id });
}

async function getCall(env: ApiEnv, userId: string, callId: string): Promise<Response> {
  const call = await env.DB.prepare(
    `SELECT id, kind, status, started_at, ended_at, duration_s FROM calls WHERE id = ?1 AND user_id = ?2`
  )
    .bind(callId, userId)
    .first();
  if (!call) return json(404, { error: "not found" });
  const { results } = await env.DB.prepare(
    `SELECT role, text FROM call_messages WHERE call_id = ?1 ORDER BY seq`
  )
    .bind(callId)
    .all();
  return json(200, { call, messages: results });
}

const STATUSES = new Set(["active", "completed", "failed", "busy", "cancelled"]);

// Updates status/duration and, when given, replaces the whole transcript.
// The client re-sends the full transcript as the call goes, so a tab that
// dies mid-call still leaves everything up to its last save.
async function updateCall(request: Request, env: ApiEnv, userId: string, callId: string): Promise<Response> {
  const body = await readJson<{
    status?: string;
    duration_s?: number;
    ended?: boolean;
    messages?: { role?: string; text?: string }[];
  }>(request);
  if (!body) return json(400, { error: "invalid json" });

  const owned = await env.DB.prepare(`SELECT 1 FROM calls WHERE id = ?1 AND user_id = ?2`)
    .bind(callId, userId)
    .first();
  if (!owned) return json(404, { error: "not found" });

  const statements: D1PreparedStatement[] = [];
  const status = body.status && STATUSES.has(body.status) ? body.status : null;
  const duration = Number.isFinite(body.duration_s) ? Math.max(0, Math.round(body.duration_s!)) : null;
  statements.push(
    env.DB.prepare(
      `UPDATE calls SET
         status = COALESCE(?1, status),
         duration_s = COALESCE(?2, duration_s),
         ended_at = CASE WHEN ?3 THEN ?4 ELSE ended_at END,
         updated_at = ?4
       WHERE id = ?5`
    ).bind(status, duration, body.ended ? 1 : 0, Date.now(), callId)
  );

  if (Array.isArray(body.messages)) {
    const messages = body.messages
      .filter((m) => (m.role === "assistant" || m.role === "user") && typeof m.text === "string" && m.text.trim())
      .slice(0, MAX_MESSAGES);
    statements.push(env.DB.prepare(`DELETE FROM call_messages WHERE call_id = ?1`).bind(callId));
    messages.forEach((m, seq) => {
      statements.push(
        env.DB.prepare(`INSERT INTO call_messages (call_id, seq, role, text) VALUES (?1, ?2, ?3, ?4)`).bind(
          callId,
          seq,
          m.role!,
          m.text!.trim().slice(0, MAX_MESSAGE_CHARS)
        )
      );
    });
  }

  await env.DB.batch(statements);
  return json(200, { ok: true });
}

async function deleteCall(env: ApiEnv, userId: string, callId: string): Promise<Response> {
  const { meta } = await env.DB.prepare(`DELETE FROM calls WHERE id = ?1 AND user_id = ?2`)
    .bind(callId, userId)
    .run();
  if (!meta.changes) return json(404, { error: "not found" });
  await env.DB.prepare(`DELETE FROM call_messages WHERE call_id = ?1`).bind(callId).run();
  return json(200, { ok: true });
}

export async function handleApi(request: Request, env: ApiEnv, url: URL): Promise<Response> {
  const path = url.pathname;
  const method = request.method;

  if (path === "/api/config" && method === "GET") {
    return json(200, {
      authEnabled: authEnabled(env),
      googleClientId: env.GOOGLE_CLIENT_ID || null,
      phoneScope: PHONE_SCOPE,
      // the UI hangs up cleanly at this limit; the container is also killed
      // server-side at the same point as a backstop
      maxCallMinutes: Number(env.MAX_CALL_MINUTES) || null,
    });
  }
  if (!authEnabled(env)) return json(404, { error: "sign-in is not configured" });

  if (path === "/api/auth/google" && method === "POST") return signIn(request, env);
  if (path === "/api/auth/logout" && method === "POST") {
    return json(200, { ok: true }, { "Set-Cookie": clearedSessionCookie() });
  }

  const userId = await sessionUserId(request, env.SESSION_SECRET);
  if (!userId) return json(401, { error: "sign in required" });

  if (path === "/api/me" && method === "GET") {
    const user = await env.DB.prepare(`SELECT ${USER_COLUMNS} FROM users WHERE id = ?1`)
      .bind(userId)
      .first<UserRow>();
    return user ? json(200, { user: publicUser(user) }) : json(401, { error: "sign in required" });
  }
  if (path === "/api/me/phone" && method === "POST") return savePhoneFromGoogle(request, env, userId);
  if (path === "/api/me/phone/skip" && method === "POST") return skipPhone(env, userId);
  if (path === "/api/calls") {
    if (method === "GET") return listCalls(env, userId);
    if (method === "POST") return createCall(request, env, userId);
  }
  const m = path.match(/^\/api\/calls\/([0-9a-f-]{36})$/);
  if (m) {
    if (method === "GET") return getCall(env, userId, m[1]);
    if (method === "PUT") return updateCall(request, env, userId, m[1]);
    if (method === "DELETE") return deleteCall(env, userId, m[1]);
  }
  return json(404, { error: "not found" });
}
