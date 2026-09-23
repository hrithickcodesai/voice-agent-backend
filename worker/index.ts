import { Container } from "@cloudflare/containers";

interface Env {
  VOICE_AGENT_CONTAINER: DurableObjectNamespace<VoiceAgentContainer>;
  ELEVENLABS_API_KEY: string;
  ELEVENLABS_VOICE_ID: string;
  OPENROUTER_API_KEY: string;
  PIPECAT_ICE_SERVERS: string;
}

export class VoiceAgentContainer extends Container<Env> {
  defaultPort = 7860;
  // Each container handles exactly one call, so it only needs to stay warm
  // long enough to survive reconnects/renegotiation after the call ends.
  sleepAfter = "3m";
  enableInternet = true;
  pingEndpoint = "/status";

  envVars = {
    ELEVENLABS_API_KEY: this.env.ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID: this.env.ELEVENLABS_VOICE_ID,
    OPENROUTER_API_KEY: this.env.OPENROUTER_API_KEY,
    PIPECAT_ICE_SERVERS: this.env.PIPECAT_ICE_SERVERS,
  };
}

const SESSION_COOKIE = "cf_call_session";

// pipecat generates its own internal session id inside whichever container
// handles /start, with no way for us to supply one - so container affinity
// can't be keyed on that id (routing /sessions/{that-id}/... via getByName
// would hash to an unrelated, empty container). Instead the Worker assigns
// its own affinity key up front via a cookie, set on the first response a
// browser tab receives (loading /ui) so it's already present by the time
// the client calls /start and every request after - one call, one container.
function readSessionCookie(request: Request): string | null {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const [name, value] = part.trim().split("=");
    if (name === SESSION_COOKIE && value) return value;
  }
  return null;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const existing = readSessionCookie(request);
    const sessionId = existing ?? crypto.randomUUID();

    const container = env.VOICE_AGENT_CONTAINER.getByName(sessionId);
    await container.startAndWaitForPorts();
    const response = await container.fetch(request);

    if (existing) return response;

    const withCookie = new Response(response.body, response);
    withCookie.headers.append(
      "Set-Cookie",
      `${SESSION_COOKIE}=${sessionId}; Path=/; Max-Age=3600; HttpOnly; Secure; SameSite=Lax`
    );
    return withCookie;
  },
};
