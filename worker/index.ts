import { Container } from "@cloudflare/containers";
import { checkAdmin, handleAdminApi, type AdminEnv } from "./admin";
import { authEnabled, handleApi, type ApiEnv } from "./api";
import { sessionUserId } from "./auth";

interface Env extends ApiEnv, AdminEnv {
  VOICE_AGENT_CONTAINER: DurableObjectNamespace<VoiceAgentContainer>;
  ASSETS: Fetcher;
  ELEVENLABS_API_KEY: string;
  ELEVENLABS_VOICE_ID: string;
  OPENROUTER_API_KEY: string;
  PIPECAT_ICE_SERVERS: string;
  LLM_MODEL: string;
  AGENT_NAME: string;
  // scaling & cost switches, see wrangler.jsonc
  CALLS_ENABLED: string;
  MAX_CALL_MINUTES: string;
  DAILY_MINUTES_PER_USER: string;
}

function minutesVar(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

export class VoiceAgentContainer extends Container<Env> {
  defaultPort = 7860;
  // Idle timeout for a /start that never turns into a call. Once a call is
  // live, its audio flows over WebRTC and never passes through this Durable
  // Object, so "idle" says nothing about the call - see onActivityExpired.
  sleepAfter = "2m";
  enableInternet = true;
  pingEndpoint = "/status";

  envVars = {
    ELEVENLABS_API_KEY: this.env.ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID: this.env.ELEVENLABS_VOICE_ID,
    OPENROUTER_API_KEY: this.env.OPENROUTER_API_KEY,
    PIPECAT_ICE_SERVERS: this.env.PIPECAT_ICE_SERVERS,
    LLM_MODEL: this.env.LLM_MODEL,
    AGENT_NAME: this.env.AGENT_NAME,
    EXIT_AFTER_CALL: "true",
  };

  private maxCallMs(): number {
    return minutesVar(this.env.MAX_CALL_MINUTES, 30) * 60_000;
  }

  override async fetch(request: Request): Promise<Response> {
    const isOffer = request.method === "POST" && new URL(request.url).pathname.endsWith("/api/offer");
    if (isOffer && !(await this.ctx.storage.get("callStartedAt"))) {
      await this.ctx.storage.put("callStartedAt", Date.now());
      // MAX_CALL_MINUTES: end the call exactly on time
      await this.schedule(Math.ceil(this.maxCallMs() / 1000), "endCallAtLimit");
    }
    return super.fetch(request);
  }

  // SIGKILL, not stop()'s SIGTERM: on SIGTERM uvicorn waits for background
  // tasks to finish, and a live call *is* one, so the container would keep
  // running until the caller hung up (verified locally).
  async endCallAtLimit(): Promise<void> {
    console.log("call reached MAX_CALL_MINUTES, destroying container");
    await this.destroy();
  }

  // The default stops the container once sleepAfter passes without a request
  // - which cut every call off ~2 minutes in, mid-conversation. After the
  // WebRTC offer, the bot owns the lifecycle instead: it exits the container
  // itself when the caller hangs up or drops (EXIT_AFTER_CALL), and
  // endCallAtLimit enforces MAX_CALL_MINUTES. This is only a backstop.
  override async onActivityExpired(): Promise<void> {
    const startedAt = await this.ctx.storage.get<number>("callStartedAt");
    if (startedAt && Date.now() - startedAt < this.maxCallMs() + 60_000) {
      this.renewActivityTimeout();
      return;
    }
    if (startedAt) {
      // past the cap and still running: force it (see endCallAtLimit)
      await this.destroy();
      return;
    }
    await super.onActivityExpired();
  }

  override async onStop(): Promise<void> {
    await this.ctx.storage.delete("callStartedAt");
  }
}

const CALL_COOKIE = "cf_call_session";

function readCallCookie(request: Request): string | null {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const [name, value] = part.trim().split("=");
    if (name === CALL_COOKIE && value) return value;
  }
  return null;
}

function json(status: number, body: object, headers: HeadersInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

async function forwardToContainer(request: Request, env: Env, callId: string): Promise<Response> {
  const container = env.VOICE_AGENT_CONTAINER.getByName(callId);
  try {
    // Cloudflare keeps a finite pool of prewarmed instances per location;
    // when simultaneous calls drain the nearby one, starts fail with "no
    // container instance that can be provided" until more are prepped a few
    // seconds later. The library's default 8s gives up too early and turned
    // a second concurrent caller into "Line Busy" - keep trying while the
    // phone rings instead.
    await container.startAndWaitForPorts({
      cancellationOptions: { instanceGetTimeoutMS: 30_000, portReadyTimeoutMS: 30_000 },
    });
    return await container.fetch(request);
  } catch (err) {
    // Most commonly max_instances reached (every line busy) or a cold
    // start that didn't come up in time - fail soft instead of a 1101.
    console.error("container start/fetch failed", callId, err);
    // "info" is the field pipecat's client surfaces as the error message;
    // the UI keys its carrier-style "Line Busy" screen off this value.
    return json(503, { info: "line_busy" }, { "Retry-After": "30" });
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // The UI is plain static files, served straight from Workers assets so
    // loading the page (or a bot/link preview hitting it) never costs a container.
    if (url.pathname === "/" || url.pathname === "/ui") {
      return Response.redirect(new URL("/ui/", url).toString(), 302);
    }
    // Admin panel: signed in with Google AND email in ADMIN_EMAILS. The page
    // itself holds no data (it all comes from /api/admin/*, which checks
    // again), but it's still only served to admins.
    if (url.pathname === "/admin" || url.pathname === "/admin/") {
      const auth = await checkAdmin(request, env);
      if (!auth.ok && auth.reason === "signed_out") {
        return Response.redirect(new URL("/ui/?next=/admin", url).toString(), 302);
      }
      if (!auth.ok) {
        return new Response("This Google account isn't an admin.", {
          status: 403,
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        });
      }
      return env.ASSETS.fetch(new Request(new URL("/admin/", url), request));
    }
    if (url.pathname.startsWith("/api/admin/")) return handleAdminApi(request, env, url);
    if (url.pathname.startsWith("/ui/admin")) return json(404, { error: "Not found" });

    if (url.pathname.startsWith("/ui/")) {
      const assetUrl = new URL(url.pathname.slice("/ui".length) + url.search, url);
      return env.ASSETS.fetch(new Request(assetUrl, request));
    }

    // sign-in and the call log (Recents)
    if (url.pathname.startsWith("/api/")) return handleApi(request, env, url);

    // Every call gets its own fresh container, keyed by a new id minted here.
    // pipecat generates its own session id inside the container with no way
    // to supply one, so the follow-up /sessions/{id}/... requests find their
    // container via this cookie instead. The bot shuts the container down
    // when the call ends, and a redial never reuses it.
    if (request.method === "POST" && url.pathname === "/start") {
      // CALLS_ENABLED kill switch
      if (env.CALLS_ENABLED === "false") return json(503, { info: "calls_paused" });
      // every call costs real money, so once sign-in is configured only
      // signed-in users can start one
      const userId = authEnabled(env) ? await sessionUserId(request, env.SESSION_SECRET) : null;
      if (authEnabled(env) && !userId) return json(401, { info: "sign_in_required" });
      // DAILY_MINUTES_PER_USER: talk time in the last 24h, including any call
      // still in progress (its duration is kept current by the heartbeat)
      const dailyLimit = minutesVar(env.DAILY_MINUTES_PER_USER, 0);
      if (userId && dailyLimit > 0) {
        const used = await env.DB.prepare(
          `SELECT COALESCE(SUM(duration_s), 0) AS s FROM calls WHERE user_id = ?1 AND started_at >= ?2`
        )
          .bind(userId, Date.now() - 86_400_000)
          .first<{ s: number }>();
        if ((used?.s ?? 0) >= dailyLimit * 60) return json(429, { info: "daily_limit" });
      }
      // Right after other calls end, Cloudflare can hand a new call an
      // instance that's still being torn down ("Container suddenly
      // disconnected, try again"). /start has no state yet, so retry it on
      // a fresh container rather than failing the call.
      const body = await request.arrayBuffer();
      let callId = "";
      let response: Response | null = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        callId = crypto.randomUUID();
        response = await forwardToContainer(new Request(request, { body }), env, callId);
        if (response.status < 500 || response.status === 503) break;
        console.warn("start failed, retrying on a fresh container", response.status, await response.text());
      }
      response = response!;
      const withCookie = new Response(response.body, response);
      withCookie.headers.append(
        "Set-Cookie",
        `${CALL_COOKIE}=${callId}; Path=/; Max-Age=3600; HttpOnly; Secure; SameSite=Lax`
      );
      return withCookie;
    }

    const callId = readCallCookie(request);
    if (!callId || !url.pathname.startsWith("/sessions/")) {
      return json(404, { error: "Not found" });
    }
    return forwardToContainer(request, env, callId);
  },
};
