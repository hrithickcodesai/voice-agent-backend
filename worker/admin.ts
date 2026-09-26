import { sessionUserId } from "./auth";

// Admin panel API. Access = signed in with Google (normal session cookie)
// AND the account email is in ADMIN_EMAILS, a comma-separated list kept by
// hand in wrangler.jsonc.

export interface AdminEnv {
  DB: D1Database;
  SESSION_SECRET: string;
  ADMIN_EMAILS: string;
}

// A call is live while its status is 'active' and the client has saved it
// (transcript or 10s heartbeat) recently.
const LIVE_WINDOW_MS = 45_000;
const HOUR = 3_600_000;
const DAY = 24 * HOUR;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function adminEmails(env: AdminEnv): Set<string> {
  return new Set(
    (env.ADMIN_EMAILS || "")
      .split(",")
      .map((e) => e.trim().toLowerCase())
      .filter(Boolean)
  );
}

export type AdminCheck = { ok: true; email: string } | { ok: false; reason: "signed_out" | "forbidden" };

export async function checkAdmin(request: Request, env: AdminEnv): Promise<AdminCheck> {
  const userId = await sessionUserId(request, env.SESSION_SECRET);
  if (!userId) return { ok: false, reason: "signed_out" };
  const user = await env.DB.prepare(`SELECT email FROM users WHERE id = ?1`)
    .bind(userId)
    .first<{ email: string }>();
  if (!user) return { ok: false, reason: "signed_out" };
  return adminEmails(env).has(user.email.toLowerCase())
    ? { ok: true, email: user.email }
    : { ok: false, reason: "forbidden" };
}

async function overview(env: AdminEnv): Promise<Response> {
  const now = Date.now();
  const since = { d: now - DAY, w: now - 7 * DAY, m: now - 30 * DAY, y: now - 365 * DAY };
  const [users, calls, unique, live] = await env.DB.batch([
    env.DB.prepare(
      `SELECT COUNT(*) AS total,
              SUM(created_at >= ?1) AS new_24h, SUM(created_at >= ?2) AS new_7d, SUM(created_at >= ?3) AS new_30d
       FROM users`
    ).bind(since.d, since.w, since.m),
    env.DB.prepare(
      `SELECT COUNT(*) AS total, COALESCE(SUM(duration_s), 0) AS seconds,
              SUM(started_at >= ?1) AS calls_24h, COALESCE(SUM(CASE WHEN started_at >= ?1 THEN duration_s END), 0) AS seconds_24h
       FROM calls`
    ).bind(since.d),
    // unique users who made at least one call in each window
    env.DB.prepare(
      `SELECT COUNT(DISTINCT CASE WHEN started_at >= ?1 THEN user_id END) AS d,
              COUNT(DISTINCT CASE WHEN started_at >= ?2 THEN user_id END) AS w,
              COUNT(DISTINCT CASE WHEN started_at >= ?3 THEN user_id END) AS m,
              COUNT(DISTINCT CASE WHEN started_at >= ?4 THEN user_id END) AS y
       FROM calls WHERE started_at >= ?4`
    ).bind(since.d, since.w, since.m, since.y),
    env.DB.prepare(`SELECT COUNT(*) AS n FROM calls WHERE status = 'active' AND updated_at >= ?1`).bind(
      now - LIVE_WINDOW_MS
    ),
  ]);
  return json(200, {
    users: users.results[0],
    calls: calls.results[0],
    uniqueUsers: unique.results[0],
    liveCalls: (live.results[0] as { n: number }).n,
  });
}

// Usage over time for the chart: unique callers, calls and minutes per
// bucket (hour for 24h, day for 7d/30d, month for 1y), in the viewer's
// timezone (tz = minutes east of UTC, from the browser).
const RANGES: Record<string, { ms: number; fmt: string }> = {
  "24h": { ms: DAY, fmt: "%Y-%m-%d %H:00" },
  "7d": { ms: 7 * DAY, fmt: "%Y-%m-%d" },
  "30d": { ms: 30 * DAY, fmt: "%Y-%m-%d" },
  "1y": { ms: 365 * DAY, fmt: "%Y-%m" },
};

async function usage(env: AdminEnv, url: URL): Promise<Response> {
  const range = RANGES[url.searchParams.get("range") ?? "7d"] ?? RANGES["7d"];
  const tz = Math.max(-840, Math.min(840, Number(url.searchParams.get("tz")) || 0));
  const { results } = await env.DB.prepare(
    `SELECT strftime(?1, started_at / 1000 + ?2 * 60, 'unixepoch') AS bucket,
            COUNT(DISTINCT user_id) AS users, COUNT(*) AS calls,
            ROUND(COALESCE(SUM(duration_s), 0) / 60.0, 1) AS minutes
     FROM calls WHERE started_at >= ?3
     GROUP BY bucket ORDER BY bucket`
  )
    .bind(range.fmt, tz, Date.now() - range.ms)
    .all();
  return json(200, { buckets: results });
}

const CALL_WITH_USER = `
  SELECT c.id, c.kind, c.status, c.started_at, c.ended_at, c.updated_at, c.duration_s,
         u.id AS user_id, u.email, u.name, u.picture,
         (SELECT COUNT(*) FROM call_messages m WHERE m.call_id = c.id) AS lines
  FROM calls c JOIN users u ON u.id = c.user_id`;

async function liveCalls(env: AdminEnv): Promise<Response> {
  const { results } = await env.DB.prepare(
    `${CALL_WITH_USER} WHERE c.status = 'active' AND c.updated_at >= ?1 ORDER BY c.started_at DESC`
  )
    .bind(Date.now() - LIVE_WINDOW_MS)
    .all();
  return json(200, { calls: results });
}

async function pastCalls(env: AdminEnv, url: URL): Promise<Response> {
  const before = Number(url.searchParams.get("before")) || Date.now() + 1;
  const userId = url.searchParams.get("user");
  const stmt = userId
    ? env.DB.prepare(`${CALL_WITH_USER} WHERE c.started_at < ?1 AND c.user_id = ?2 ORDER BY c.started_at DESC LIMIT 50`).bind(
        before,
        userId
      )
    : env.DB.prepare(`${CALL_WITH_USER} WHERE c.started_at < ?1 ORDER BY c.started_at DESC LIMIT 50`).bind(before);
  const { results } = await stmt.all();
  return json(200, { calls: results, liveWindowMs: LIVE_WINDOW_MS });
}

async function callDetail(env: AdminEnv, callId: string): Promise<Response> {
  const call = await env.DB.prepare(`${CALL_WITH_USER} WHERE c.id = ?1`).bind(callId).first();
  if (!call) return json(404, { error: "not found" });
  const { results } = await env.DB.prepare(`SELECT role, text FROM call_messages WHERE call_id = ?1 ORDER BY seq`)
    .bind(callId)
    .all();
  return json(200, { call, messages: results, liveWindowMs: LIVE_WINDOW_MS });
}

async function listUsers(env: AdminEnv): Promise<Response> {
  const { results } = await env.DB.prepare(
    `SELECT u.id, u.email, u.name, u.picture, u.phone_number, u.locale, u.created_at, u.last_login_at,
            COUNT(c.id) AS calls, COALESCE(SUM(c.duration_s), 0) AS seconds, MAX(c.started_at) AS last_call_at
     FROM users u LEFT JOIN calls c ON c.user_id = u.id
     GROUP BY u.id ORDER BY u.created_at DESC LIMIT 1000`
  ).all();
  return json(200, { users: results });
}

export async function handleAdminApi(request: Request, env: AdminEnv, url: URL): Promise<Response> {
  const auth = await checkAdmin(request, env);
  if (!auth.ok) return json(auth.reason === "signed_out" ? 401 : 403, { error: auth.reason });
  if (request.method !== "GET") return json(405, { error: "method not allowed" });

  const path = url.pathname.replace(/^\/api\/admin/, "");
  if (path === "/me") return json(200, { email: auth.email });
  if (path === "/overview") return overview(env);
  if (path === "/usage") return usage(env, url);
  if (path === "/live") return liveCalls(env);
  if (path === "/calls") return pastCalls(env, url);
  if (path === "/users") return listUsers(env);
  const m = path.match(/^\/calls\/([0-9a-f-]{36})$/);
  if (m) return callDetail(env, m[1]);
  return json(404, { error: "not found" });
}
