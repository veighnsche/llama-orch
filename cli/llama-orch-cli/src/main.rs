use clap::Parser;

/// llama-orch-cli — contract/dev oriented CLI (scaffold)
#[derive(Debug, Parser)]
#[command(name = "llama-orch", version, about = "CLI frontend for llama-orch (scaffold)")]
struct Args {
    /// Orchestrator address (host:port)
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
    /// Auth token (will also read AUTH_TOKEN env). Never prints full token.
    #[arg(long)]
    auth_token: Option<String>,
}

fn main() {
    let args = Args::parse();
    let addr = std::env::var("ORCHD_ADDR").unwrap_or(args.addr);
    let token = args.auth_token.or_else(|| std::env::var("AUTH_TOKEN").ok());
    let token_fp = token.as_ref().map(|t| auth_min::token_fp6(t));
    println!("addr={} auth={}",
        addr,
        match token_fp {
            Some(fp) => format!("token:****{}", fp),
            None => "<none>".to_string(),
        }
    );
}
