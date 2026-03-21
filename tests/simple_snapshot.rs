use mybtree::{Database, DbValue, Pager, Row};
use std::fs;
use tempfile::NamedTempFile;

/// Helper to format a Row for display
fn format_row(row: &Row) -> String {
    row.values
        .iter()
        .map(|v| match v {
            DbValue::Null => "NULL".to_string(),
            DbValue::Integer(i) => i.to_string(),
            DbValue::Float(f) => f.to_string(),
            DbValue::Text(s) => s.clone(),
            DbValue::Bool(b) => b.to_string(),
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Split SQL script into individual statements, handling comments and whitespace
fn split_statements(sql: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();

    for line in sql.lines() {
        // Skip comment lines
        let trimmed = line.trim();
        if trimmed.starts_with("--") || trimmed.is_empty() {
            continue;
        }

        current.push_str(line);
        current.push('\n');

        // Check if statement ends with semicolon
        if trimmed.ends_with(';') {
            // Remove the trailing semicolon and trim
            let stmt = current.trim().trim_end_matches(';').to_string();
            if !stmt.is_empty() {
                statements.push(stmt);
            }
            current.clear();
        }
    }

    // Add any remaining statement
    let stmt = current.trim().to_string();
    if !stmt.is_empty() {
        statements.push(stmt);
    }

    statements
}

/// Execute a SQL file and return formatted output for snapshotting
fn execute_sql_file(sql_path: &str) -> String {
    // Read the SQL file
    let sql_content =
        fs::read_to_string(sql_path).unwrap_or_else(|_| panic!("Failed to read {}", sql_path));

    // Create a temporary database
    let temp = NamedTempFile::new().unwrap();
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(temp.path())
        .unwrap();
    let pager = Pager::<4096>::new(file);
    let db = Database::create(pager).unwrap();

    // Split and execute statements
    let statements = split_statements(&sql_content);
    let mut output = String::new();
    let mut tx = None;

    for (idx, stmt) in statements.iter().enumerate() {
        output.push_str(&format!("-- Statement {}\n", idx + 1));
        output.push_str(&format!("-- {}\n", stmt.replace('\n', " ")));

        match mybtree::sql::execute(&db, &mut tx, stmt) {
            Ok(rows) => {
                if rows.is_empty() {
                    output.push_str("-- OK (no results)\n");
                } else {
                    for row in rows {
                        output.push_str(&format!("-- {}\n", format_row(&row)));
                    }
                }
            }
            Err(e) => {
                output.push_str(&format!("-- ERROR: {}\n", e));
            }
        }

        output.push('\n');
    }

    output
}

#[test]
fn test_simple_sql_snapshot() {
    let output = execute_sql_file("tests/sql/simple.sql");
    insta::assert_snapshot!("simple_sql_output", output);
}

#[test]
fn test_index_sql_snapshot() {
    let output = execute_sql_file("tests/sql/index_test.sql");
    insta::assert_snapshot!("index_sql_output", output);
}
