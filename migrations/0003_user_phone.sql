-- Phone numbers aren't in Google's ID token; they come from the People API
-- after the user grants the extra user.phonenumbers.read permission, which
-- we ask for once, optionally, right after signup.

-- the primary number (E.164 when Google provides a canonical form)
ALTER TABLE users ADD COLUMN phone_number TEXT;
-- every phoneNumbers entry the People API returned, as JSON
ALTER TABLE users ADD COLUMN phone_numbers TEXT;
-- when we asked (granted, declined or skipped) - so we only ask once
ALTER TABLE users ADD COLUMN phone_prompted_at INTEGER;
