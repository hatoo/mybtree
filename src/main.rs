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
    /// Path to the database file (required unless --tmp is used)
    #[arg(value_name = "DB_PATH", required_unless_present = "tmp")]
    db: Option<PathBuf>,

    /// Use a temporary database file in the system temp directory
    #[arg(long)]
    tmp: bool,

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

fn resolve_db_path(path: Option<&PathBuf>, tmp: bool) -> anyhow::Result<PathBuf> {
    if tmp {
        let temp_dir = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_nanos();
        let db_name = format!("mybtree_{}.db", timestamp);
        return Ok(temp_dir.join(db_name));
    }

    path.cloned()
        .ok_or_else(|| anyhow::anyhow!("database path is required unless --tmp is used"))
}

fn open_db(path: Option<&PathBuf>, tmp: bool) -> anyhow::Result<Database<PAGE_SIZE>> {
    let resolved_path = resolve_db_path(path, tmp)?;
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

fn execute_and_print<'a>(
    db: &'a Database<PAGE_SIZE>,
    tx: &mut Option<mybtree::DbTransaction<'a, PAGE_SIZE>>,
    input: &str,
) -> anyhow::Result<()> {
    if input == ".tables" {
        let mut t = db.begin_transaction();
        for name in t.with_lock(|mut l| l.list_tables())? {
            println!("{name}");
        }
        return Ok(());
    }

    let rs = mybtree::sql::execute(db, tx, input)?;

    for row in &rs.rows {
        let line: Vec<String> = row.iter().map(|(_, v)| format_value(v)).collect();
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

    let mut tx = None;
    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let sql = line.trim();
                if sql.is_empty() {
                    continue;
                }
                if let Err(e) = execute_and_print(db, &mut tx, sql) {
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
    let mut tx = None;

    // Split by semicolons and execute each statement
    for statement in content.split(';') {
        // Remove comments (lines starting with --)
        let lines: Vec<&str> = statement
            .lines()
            .filter(|line| !line.trim().starts_with("--"))
            .collect();

        let cleaned = lines.join(" ").trim().to_string();

        if !cleaned.is_empty() {
            execute_and_print(db, &mut tx, &cleaned)?;
        }
    }

    Ok(())
}

fn run(cli: Cli) -> anyhow::Result<()> {
    let Cli { db, tmp, sql } = cli;

    // With positional parsing, `mybtree --tmp "SQL"` is parsed as db=<SQL>, sql=None.
    // Normalize that shape so tmp mode still accepts one positional SQL argument.
    let sql = if tmp {
        match (db.as_ref(), sql) {
            (Some(_), Some(_)) => {
                anyhow::bail!("when using --tmp, provide at most one positional SQL argument");
            }
            (Some(sql_like), None) => Some(sql_like.to_string_lossy().into_owned()),
            (None, sql) => sql,
        }
    } else {
        sql
    };

    let db = open_db(db.as_ref(), tmp)?;

    match sql {
        Some(sql) => {
            // Check if sql is a file path
            if PathBuf::from(&sql).exists() && sql.ends_with(".sql") {
                execute_sql_file(&db, &sql)?;
            } else {
                execute_and_print(&db, &mut None, &sql)?;
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
