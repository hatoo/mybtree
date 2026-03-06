-- table
CREATE TABLE users(
  id INTEGER,
  name TEXT,
  age INTEGER
);

-- data
INSERT INTO users VALUES (1, 'alice', 20);
INSERT INTO users VALUES (2, 'bob', 25);
INSERT INTO users VALUES (3, 'carol', 30);
INSERT INTO users VALUES (4, 'dave', 35);
INSERT INTO users VALUES (5, 'eve', 40);

-- index
CREATE INDEX idx_users_id ON users(id);

-- index lookup
SELECT * FROM users WHERE id = 3;

-- range scan
SELECT * FROM users WHERE id >= 2 AND id <= 4;

-- projection
SELECT name FROM users WHERE id = 5;

-- update indexed column
UPDATE users
SET id = 6
WHERE id = 5;

-- verify update
SELECT * FROM users WHERE id = 6;

-- delete indexed row
DELETE FROM users
WHERE id = 1;

-- verify delete
SELECT * FROM users;