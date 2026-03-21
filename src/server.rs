use std::sync::Mutex;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use clap::Parser;
use mybtree::{Database, DbTransaction, DbValue, Pager, Row};
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
#[serde(untagged)]
enum JsonValue {
    Integer(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Null,
}

impl From<&DbValue> for JsonValue {
    fn from(v: &DbValue) -> Self {
        match v {
            DbValue::Integer(i) => JsonValue::Integer(*i),
            DbValue::Float(f) => JsonValue::Float(*f),
            DbValue::Bool(b) => JsonValue::Bool(*b),
            DbValue::Text(s) => JsonValue::Text(s.clone()),
            DbValue::Null => JsonValue::Null,
        }
    }
}

#[derive(Serialize)]
struct SqlResponse {
    rows: Vec<Vec<JsonValue>>,
}

fn rows_to_response(rows: Vec<Row>) -> SqlResponse {
    SqlResponse {
        rows: rows
            .iter()
            .map(|r| r.values.iter().map(JsonValue::from).collect())
            .collect(),
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
        Ok(rows) => Ok(Json(rows_to_response(rows))),
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
