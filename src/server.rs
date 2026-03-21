use std::sync::Mutex;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use clap::Parser;
use mybtree::{Database, DbTransaction, DbValue, Pager};
use serde::{Deserialize, Serialize};

const PAGE_SIZE: usize = 4096;

struct AppState {
    db: Database<PAGE_SIZE>,
    tx: Mutex<Option<DbTransaction<'static, PAGE_SIZE>>>,
}

#[derive(Deserialize)]
struct SqlRequest {
    sql: String,
}

#[derive(Serialize)]
struct SqlResponse {
    rows: Vec<serde_json::Map<String, serde_json::Value>>,
}

fn to_json_value(v: &DbValue) -> serde_json::Value {
    match v {
        DbValue::Integer(i) => serde_json::Value::Number((*i).into()),
        DbValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        DbValue::Bool(b) => serde_json::Value::Bool(*b),
        DbValue::Text(s) => serde_json::Value::String(s.clone()),
        DbValue::Null => serde_json::Value::Null,
    }
}

async fn execute_sql(
    State(state): State<&'static AppState>,
    Json(req): Json<SqlRequest>,
) -> Result<Json<SqlResponse>, (StatusCode, String)> {
    let result = {
        let mut tx = state.tx.lock().unwrap();
        mybtree::sql::execute(&state.db, &mut tx, &req.sql)
    };

    match result {
        Ok(rs) => {
            let rows = rs
                .rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|(name, val)| (name.clone(), to_json_value(val)))
                        .collect()
                })
                .collect();
            Ok(Json(SqlResponse { rows }))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

#[derive(Parser)]
#[command(about = "mybtree SQL server")]
struct Cli {
    /// Database file path
    #[arg(default_value = "mybtree.db")]
    db: String,

    /// Address to listen on
    #[arg(short, long, default_value = "0.0.0.0:3000")]
    listen: String,
}

fn file_is_empty(path: &str) -> bool {
    std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&cli.db)
        .expect("failed to open database file");

    let pager = Pager::<PAGE_SIZE>::new(file);
    let db = if file_is_empty(&cli.db) {
        Database::create(pager).expect("failed to create database")
    } else {
        Database::open(pager).expect("failed to open database")
    };

    let state: &'static AppState = Box::leak(Box::new(AppState {
        db,
        tx: Mutex::new(None),
    }));

    let app = Router::new()
        .route("/sql", post(execute_sql))
        .with_state(state);

    println!("listening on {}", cli.listen);
    let listener = tokio::net::TcpListener::bind(&cli.listen).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
