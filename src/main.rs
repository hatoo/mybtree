use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use mybtree::{Database, DbValue, Pager};
use reedline::{DefaultPrompt, DefaultPromptSegment, Reedline, Signal};

const PAGE_SIZE: usize = 4096;

#[derive(Parser)]
#[command(name = "mybtree", about = "Execute SQL operations on a database file")]
struct Cli {
    /// Path to the database file
    db: PathBuf,

    /// SQL statement to execute (omit to enter REPL)
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

fn open_db(path: &PathBuf) -> anyhow::Result<Database> {
    let create = !path.exists();
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;

    let pager = Pager::new(file, PAGE_SIZE);
    if create {
        Ok(Database::create(pager)?)
    } else {
        Ok(Database::open(pager)?)
    }
}

fn execute_and_print(db: &Database, input: &str) -> anyhow::Result<()> {
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

fn repl(db: &Database) -> anyhow::Result<()> {
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

fn run(cli: Cli) -> anyhow::Result<()> {
    let db = open_db(&cli.db)?;

    match cli.sql {
        Some(sql) => execute_and_print(&db, &sql),
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
