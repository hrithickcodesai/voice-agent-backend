-- Signed-in users (Google accounts) and their call log (iOS-style Recents).

CREATE TABLE users (
  id TEXT PRIMARY KEY,
  google_sub TEXT NOT NULL UNIQUE,
  email TEXT NOT NULL,
  name TEXT,
  picture TEXT,
  created_at INTEGER NOT NULL,
  last_login_at INTEGER NOT NULL
);

CREATE TABLE calls (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('audio', 'facetime')),
  -- 'active' while in progress; 'completed' | 'failed' | 'busy' | 'cancelled' after
  status TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  duration_s INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX calls_by_user ON calls (user_id, started_at DESC);

CREATE TABLE call_messages (
  call_id TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('assistant', 'user')),
  text TEXT NOT NULL,
  PRIMARY KEY (call_id, seq)
);
