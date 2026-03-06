-- create table
CREATE TABLE users(
  id INTEGER,
  name TEXT,
  age INTEGER
);

-- insert rows
INSERT INTO users VALUES (1, 'alice', 20);
INSERT INTO users VALUES (2, 'bob', 25);
INSERT INTO users VALUES (3, 'carol', 30);

-- full scan
SELECT * FROM users;

-- simple predicate
SELECT * FROM users WHERE id = 2;

-- projection
SELECT name FROM users;

-- update
UPDATE users
SET age = 26
WHERE id = 2;

-- check update
SELECT * FROM users WHERE id = 2;

-- delete
DELETE FROM users
WHERE id = 1;

-- check delete
SELECT * FROM users;