use std::fs;

use tempfile::NamedTempFile;

use crate::{Database, DbValue, Pager};

use super::{ResultSet, execute};

fn open_db() -> (Database<4096>, NamedTempFile) {
    let temp = NamedTempFile::new().unwrap();
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(temp.path())
        .unwrap();
    let pager = Pager::<4096>::new(file);
    let db = Database::create(pager).unwrap();
    (db, temp)
}

/// Helper to extract just the values from a ResultSet for easy assertion.
fn values(rs: &ResultSet) -> Vec<Vec<DbValue>> {
    rs.rows
        .iter()
        .map(|r| r.iter().map(|(_, v)| v.clone()).collect())
        .collect()
}

#[test]
fn create_insert_select() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    )
    .unwrap();

    let rs = execute(&db, &mut None, "SELECT name, age FROM users").unwrap();

    assert_eq!(rs.rows.len(), 2);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
    assert_eq!(rs.rows[0][1], ("age".into(), DbValue::Integer(30)));
    assert_eq!(
        values(&rs),
        vec![
            vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
            vec![DbValue::Text("Bob".into()), DbValue::Integer(25)],
        ]
    );
}

#[test]
fn select_with_alias() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();

    let rs = execute(&db, &mut None, "SELECT name AS user_name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("user_name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn update_rows() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    )
    .unwrap();

    execute(&db, &mut None, "UPDATE users SET age = 31 WHERE id = 1").unwrap();

    let rs = execute(&db, &mut None, "SELECT name, age FROM users").unwrap();
    assert_eq!(
        values(&rs),
        vec![
            vec![DbValue::Text("Alice".into()), DbValue::Integer(31)],
            vec![DbValue::Text("Bob".into()), DbValue::Integer(25)],
        ]
    );

    execute(&db, &mut None, "UPDATE users SET name = 'Unknown'").unwrap();

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(
        values(&rs),
        vec![
            vec![DbValue::Text("Unknown".into())],
            vec![DbValue::Text("Unknown".into())]
        ]
    );
}

#[test]
fn delete_rows() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)",
    )
    .unwrap();

    execute(&db, &mut None, "DELETE FROM users WHERE id = 1").unwrap();

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert_eq!(
        values(&rs),
        vec![
            vec![DbValue::Text("Bob".into())],
            vec![DbValue::Text("Carol".into())]
        ]
    );

    execute(&db, &mut None, "DELETE FROM users").unwrap();

    let rs = execute(&db, &mut None, "SELECT * FROM users").unwrap();
    assert_eq!(rs.rows.len(), 0);
}

#[test]
fn begin_commit() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "BEGIN;
         CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
         INSERT INTO users (id, name) VALUES (1, 'Alice');
         COMMIT",
    )
    .unwrap();

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn rollback() {
    let (db, _temp) = open_db();

    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
         INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();

    execute(
        &db,
        &mut None,
        "BEGIN;
         INSERT INTO users (id, name) VALUES (2, 'Bob');
         ROLLBACK",
    )
    .unwrap();

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn partial_transaction_across_calls() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();

    let mut tx = None;
    execute(&db, &mut tx, "BEGIN").unwrap();
    execute(
        &db,
        &mut tx,
        "INSERT INTO users (id, name) VALUES (2, 'Bob')",
    )
    .unwrap();

    let rs = execute(&db, &mut tx, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 2);

    execute(&db, &mut tx, "COMMIT").unwrap();
    assert!(tx.is_none());

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 2);
}

#[test]
fn partial_transaction_rollback_across_calls() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();

    let mut tx = None;
    execute(&db, &mut tx, "BEGIN").unwrap();
    execute(
        &db,
        &mut tx,
        "INSERT INTO users (id, name) VALUES (2, 'Bob')",
    )
    .unwrap();
    execute(&db, &mut tx, "ROLLBACK").unwrap();
    assert!(tx.is_none());

    let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn table_alias() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();

    let rs = execute(&db, &mut None, "SELECT u.name FROM users AS u").unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );

    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name FROM users AS u WHERE u.id = 1",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );

    let rs = execute(&db, &mut None, "SELECT u.name AS user_name FROM users AS u").unwrap();
    assert_eq!(
        rs.rows[0][0],
        ("user_name".into(), DbValue::Text("Alice".into()))
    );

    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (2, 'Bob')",
    )
    .unwrap();
    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name FROM users AS u WHERE u.id >= 1",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
}

#[test]
fn subquery() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)",
    )
    .unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT t.name FROM (SELECT name, age FROM users WHERE age > 28) AS t",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
    assert_eq!(
        rs.rows[1][0],
        ("name".into(), DbValue::Text("Carol".into()))
    );

    let rs = execute(
        &db,
        &mut None,
        "SELECT t.name FROM (SELECT name, age FROM users) AS t WHERE t.age > 28",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
    assert_eq!(
        rs.rows[1][0],
        ("name".into(), DbValue::Text("Carol".into()))
    );

    let rs = execute(
        &db,
        &mut None,
        "SELECT * FROM (SELECT name FROM users WHERE id = 1) AS t",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn exists_uncorrelated() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO orders (id, user_id) VALUES (1, 1)",
    )
    .unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT name FROM users WHERE EXISTS (SELECT id FROM orders)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );

    let rs = execute(
        &db,
        &mut None,
        "SELECT name FROM users WHERE NOT EXISTS (SELECT id FROM orders)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 0);
}

#[test]
fn exists_correlated() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (2, 'Bob')",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO orders (id, user_id) VALUES (1, 1)",
    )
    .unwrap();

    let rs = execute(
        &db, &mut None,
        "SELECT u.name FROM users AS u WHERE EXISTS (SELECT o.id FROM orders AS o WHERE o.user_id = u.id)",
    ).unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );

    let rs = execute(
        &db, &mut None,
        "SELECT u.name FROM users AS u WHERE NOT EXISTS (SELECT o.id FROM orders AS o WHERE o.user_id = u.id)",
    ).unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Bob".into())));
}

#[test]
fn in_subquery() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (1, 'Alice')",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name) VALUES (2, 'Bob')",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO orders (id, user_id) VALUES (1, 1)",
    )
    .unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );

    let rs = execute(
        &db,
        &mut None,
        "SELECT name FROM users WHERE id NOT IN (SELECT user_id FROM orders)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Bob".into())));
}

#[test]
fn scalar_subquery() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    )
    .unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT name FROM users WHERE age = (SELECT age FROM users WHERE id = 1)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(
        rs.rows[0][0],
        ("name".into(), DbValue::Text("Alice".into()))
    );
}

#[test]
fn any_some_all() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER NOT NULL)",
    )
    .unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, age) VALUES (1, 20)").unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, age) VALUES (2, 30)").unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, age) VALUES (3, 40)").unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT id FROM users WHERE age > ANY(SELECT age FROM users WHERE id <= 2)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert_eq!(rs.rows[0][0], ("id".into(), DbValue::Integer(2)));
    assert_eq!(rs.rows[1][0], ("id".into(), DbValue::Integer(3)));

    let rs = execute(
        &db,
        &mut None,
        "SELECT id FROM users WHERE age > SOME(SELECT age FROM users WHERE id <= 2)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);

    let rs = execute(
        &db,
        &mut None,
        "SELECT id FROM users WHERE age > ALL(SELECT age FROM users WHERE id <= 2)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][0], ("id".into(), DbValue::Integer(3)));

    let rs = execute(
        &db,
        &mut None,
        "SELECT id FROM users WHERE age > ALL(SELECT age FROM users WHERE id = 999)",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 3);
}

#[test]
fn inner_join() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, item TEXT NOT NULL)",
    )
    .unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (2, 'Bob')").unwrap();
    execute(&db, &mut None, "INSERT INTO orders (id, user_id, item) VALUES (1, 1, 'Book')").unwrap();
    execute(&db, &mut None, "INSERT INTO orders (id, user_id, item) VALUES (2, 1, 'Pen')").unwrap();

    // INNER JOIN — only Alice has orders
    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name, o.item FROM users AS u INNER JOIN orders AS o ON o.user_id = u.id",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    assert_eq!(rs.rows[0][1], ("item".into(), DbValue::Text("Book".into())));
    assert_eq!(rs.rows[1][0], ("name".into(), DbValue::Text("Alice".into())));
    assert_eq!(rs.rows[1][1], ("item".into(), DbValue::Text("Pen".into())));

    // JOIN (without INNER keyword) works the same
    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name, o.item FROM users AS u JOIN orders AS o ON o.user_id = u.id",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);

    // SELECT * returns columns from both tables
    let rs = execute(
        &db,
        &mut None,
        "SELECT * FROM users AS u INNER JOIN orders AS o ON o.user_id = u.id WHERE u.id = 1",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 2);
    assert!(rs.rows[0].len() >= 4);
}

#[test]
fn inner_join_no_match() {
    let (db, _temp) = open_db();
    execute(
        &db,
        &mut None,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    .unwrap();
    execute(
        &db,
        &mut None,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL)",
    )
    .unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name FROM users AS u INNER JOIN orders AS o ON o.user_id = u.id",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 0);
}

#[test]
fn multi_join() {
    let (db, _temp) = open_db();
    execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
    execute(&db, &mut None, "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL)").unwrap();
    execute(&db, &mut None, "CREATE TABLE items (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL, name TEXT NOT NULL)").unwrap();
    execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
    execute(&db, &mut None, "INSERT INTO orders (id, user_id) VALUES (1, 1)").unwrap();
    execute(&db, &mut None, "INSERT INTO items (id, order_id, name) VALUES (1, 1, 'Book')").unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT u.name, i.name FROM users AS u \
         INNER JOIN orders AS o ON o.user_id = u.id \
         INNER JOIN items AS i ON i.order_id = o.id",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    assert_eq!(rs.rows[0][1], ("name".into(), DbValue::Text("Book".into())));
}

#[test]
fn cross_join() {
    let (db, _temp) = open_db();
    execute(&db, &mut None, "CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER NOT NULL)").unwrap();
    execute(&db, &mut None, "CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER NOT NULL)").unwrap();
    execute(&db, &mut None, "INSERT INTO a (id, x) VALUES (1, 10)").unwrap();
    execute(&db, &mut None, "INSERT INTO a (id, x) VALUES (2, 20)").unwrap();
    execute(&db, &mut None, "INSERT INTO b (id, y) VALUES (1, 100)").unwrap();
    execute(&db, &mut None, "INSERT INTO b (id, y) VALUES (2, 200)").unwrap();

    let rs = execute(
        &db,
        &mut None,
        "SELECT a.x, b.y FROM a CROSS JOIN b",
    )
    .unwrap();
    assert_eq!(rs.rows.len(), 4);
}
