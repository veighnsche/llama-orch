#[tokio::main]
async fn main() {
    orchestratord::app::bootstrap::init_observability();
    let app = orchestratord::app::bootstrap::build_app();
    let addr = std::env::var("ORCHD_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind ORCHD_ADDR");
    eprintln!("orchestratord listening on {}", addr);
    axum::serve(listener, app).await.unwrap();
}
