-- Keep everything Google's ID token tells us about a user, not just the
-- basics: the standard profile claims as columns, plus the full verified
-- claim set as JSON - once as it was at signup (never overwritten) and once
-- as of the latest sign-in.

ALTER TABLE users ADD COLUMN given_name TEXT;
ALTER TABLE users ADD COLUMN family_name TEXT;
ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN locale TEXT;
-- Google Workspace domain, only present for company/school accounts
ALTER TABLE users ADD COLUMN hosted_domain TEXT;
ALTER TABLE users ADD COLUMN signup_claims TEXT;
ALTER TABLE users ADD COLUMN google_claims TEXT;
