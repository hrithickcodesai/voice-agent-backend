import { Container } from "@cloudflare/containers";

interface Env {
  VOICE_AGENT_CONTAINER: DurableObjectNamespace<VoiceAgentContainer>;
  ASSETS: Fetcher;
  ELEVENLABS_API_KEY: string;
  ELEVENLABS_VOICE_ID: string;
  OPENROUTER_API_KEY: string;
  PIPECAT_ICE_SERVERS: string;
}

export class VoiceAgentContainer extends Container<Env> {
  defaultPort = 7860;
  // The bot exits the container itself as soon as the call ends
  // (EXIT_AFTER_CALL); this is only a fallback for a /start that never
  // turns into a connected call.
  sleepAfter = "2m";
  enableInternet = true;
  pingEndpoint = "/status";

  envVars = {
    ELEVENLABS_API_KEY: this.env.ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID: this.env.ELEVENLABS_VOICE_ID,
    OPENROUTER_API_KEY: this.env.OPENROUTER_API_KEY,
    PIPECAT_ICE_SERVERS: this.env.PIPECAT_ICE_SERVERS,
    EXIT_AFTER_CALL: "true",
  };
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
    await container.startAndWaitForPorts();
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
    if (url.pathname.startsWith("/ui/")) {
      const assetUrl = new URL(url.pathname.slice("/ui".length) + url.search, url);
      return env.ASSETS.fetch(new Request(assetUrl, request));
    }

    // Every call gets its own fresh container, keyed by a new id minted here.
    // pipecat generates its own session id inside the container with no way
    // to supply one, so the follow-up /sessions/{id}/... requests find their
    // container via this cookie instead. The bot shuts the container down
    // when the call ends, and a redial never reuses it.
    if (request.method === "POST" && url.pathname === "/start") {
      const callId = crypto.randomUUID();
      const response = await forwardToContainer(request, env, callId);
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
