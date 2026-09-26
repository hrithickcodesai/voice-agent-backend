-- Last time the client saved a call (transcript or heartbeat). A call is
-- "live" only while status is 'active' AND this is recent, so a tab killed
-- mid-call doesn't show as active forever in the admin panel.
ALTER TABLE calls ADD COLUMN updated_at INTEGER;
UPDATE calls SET updated_at = COALESCE(ended_at, started_at);
CREATE INDEX calls_by_started ON calls (started_at DESC);
