use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::SystemTime;

use clap::Parser;
use mybtree::{Database, DbValue, Pager};
use reedline::{DefaultPrompt, DefaultPromptSegment, Reedline, Signal};

const PAGE_SIZE: usize = 4096;

#[derive(Parser)]
#[command(name = "mybtree", about = "Execute SQL operations on a database file")]
struct Cli {
    /// Path to the database file (use "tmp" for a temporary database)
    db: PathBuf,

    /// SQL statement to execute, or path to SQL file (omit to enter REPL)
    sql: Option<String>,
}

fn format_value(v: &DbValue) -> String {
    match v {
        DbValue::Integer(i) => i.to_string(),
        DbValue::Text(s) => s.clone(),
        DbValue::Float(f) => f.to_string(),
        DbValue::Bool(b) => b.to_string(),
        DbValue::Null => "NULL".to_string(),
    }
}

fn resolve_db_path(path: &PathBuf) -> anyhow::Result<PathBuf> {
    let path_str = path.to_string_lossy();
    
    // Handle special "tmp" keyword for temporary database
    if path_str == "tmp" {
        let temp_dir = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_nanos();
        let db_name = format!("mybtree_{}.db", timestamp);
        return Ok(temp_dir.join(db_name));
    }
    
    Ok(path.clone())
}

fn open_db(path: &PathBuf) -> anyhow::Result<Database<PAGE_SIZE>> {
    let resolved_path = resolve_db_path(path)?;
    let create = !resolved_path.exists();
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&resolved_path)?;

    let pager = Pager::<PAGE_SIZE>::new(file);
    if create {
        Ok(Database::create(pager)?)
    } else {
        Ok(Database::open(pager)?)
    }
}

fn execute_and_print(db: &Database<PAGE_SIZE>, input: &str) -> anyhow::Result<()> {
    if input == ".tables" {
        let tx = db.begin_transaction();
        for name in tx.list_tables()? {
            println!("{name}");
        }
        return Ok(());
    }

    let tx = db.begin_transaction();
    let rows = mybtree::sql::execute(&tx, input)?;
    tx.commit()?;

    for row in &rows {
        let line: Vec<String> = row.values.iter().map(format_value).collect();
        println!("{}", line.join("\t"));
    }

    Ok(())
}

fn repl(db: &Database<PAGE_SIZE>) -> anyhow::Result<()> {
    let mut line_editor = Reedline::create();
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("mybtree".to_string()),
        DefaultPromptSegment::Empty,
    );

    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let sql = line.trim();
                if sql.is_empty() {
                    continue;
                }
                if let Err(e) = execute_and_print(db, sql) {
                    eprintln!("error: {e}");
                }
            }
            Ok(Signal::CtrlD | Signal::CtrlC) => {
                break;
            }
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        }
    }

    Ok(())
}

fn execute_sql_file(db: &Database<PAGE_SIZE>, path: &str) -> anyhow::Result<()> {
    let content = fs::read_to_string(path)?;

    // Split by semicolons and execute each statement
    for statement in content.split(';') {
        // Remove comments (lines starting with --)
        let lines: Vec<&str> = statement
            .lines()
            .filter(|line| !line.trim().starts_with("--"))
            .collect();

        let cleaned = lines.join(" ").trim().to_string();

        if !cleaned.is_empty() {
            execute_and_print(db, &cleaned)?;
        }
    }

    Ok(())
}

fn run(cli: Cli) -> anyhow::Result<()> {
    let db = open_db(&cli.db)?;

    match cli.sql {
        Some(sql) => {
            // Check if sql is a file path
            if PathBuf::from(&sql).exists() && sql.ends_with(".sql") {
                execute_sql_file(&db, &sql)?;
            } else {
                execute_and_print(&db, &sql)?;
            }
            Ok(())
        }
        None => repl(&db),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
