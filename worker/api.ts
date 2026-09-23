import {
  clearedSessionCookie,
  createSessionToken,
  sessionCookie,
  sessionUserId,
  verifyGoogleCredential,
} from "./auth";

export interface ApiEnv {
  DB: D1Database;
  GOOGLE_CLIENT_ID: string;
  SESSION_SECRET: string;
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

  const now = Date.now();
  const user = await env.DB.prepare(
    `INSERT INTO users (id, google_sub, email, name, picture, created_at, last_login_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
     ON CONFLICT (google_sub) DO UPDATE SET
       email = excluded.email, name = excluded.name, picture = excluded.picture,
       last_login_at = excluded.last_login_at
     RETURNING id, email, name, picture`
  )
    .bind(crypto.randomUUID(), profile.sub, profile.email, profile.name ?? null, profile.picture ?? null, now)
    .first<UserRow>();

  const token = await createSessionToken(user!.id, env.SESSION_SECRET);
  return json(200, { user }, { "Set-Cookie": sessionCookie(token) });
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
    `INSERT INTO calls (id, user_id, kind, status, started_at) VALUES (?1, ?2, ?3, 'active', ?4)`
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
         ended_at = CASE WHEN ?3 THEN ?4 ELSE ended_at END
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
    return json(200, { authEnabled: authEnabled(env), googleClientId: env.GOOGLE_CLIENT_ID || null });
  }
  if (!authEnabled(env)) return json(404, { error: "sign-in is not configured" });

  if (path === "/api/auth/google" && method === "POST") return signIn(request, env);
  if (path === "/api/auth/logout" && method === "POST") {
    return json(200, { ok: true }, { "Set-Cookie": clearedSessionCookie() });
  }

  const userId = await sessionUserId(request, env.SESSION_SECRET);
  if (!userId) return json(401, { error: "sign in required" });

  if (path === "/api/me" && method === "GET") {
    const user = await env.DB.prepare(`SELECT id, email, name, picture FROM users WHERE id = ?1`)
      .bind(userId)
      .first<UserRow>();
    return user ? json(200, { user }) : json(401, { error: "sign in required" });
  }
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
