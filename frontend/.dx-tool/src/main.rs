// Created by: TEAM-DX-001
// Frontend DX CLI Tool - Main entry point

use clap::{Parser, Subcommand};
use dx::commands::CssCommand;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "dx")]
#[command(version)]
#[command(about = "Frontend DX CLI Tool - Verify CSS/HTML without browser access", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// CSS verification commands
    Css {
        /// Check if a CSS class exists in stylesheets
        #[arg(long, value_name = "CLASS")]
        class_exists: Option<String>,
        
        /// URL to fetch and analyze
        url: String,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Css { class_exists, url } => {
            if let Some(class_name) = class_exists {
                handle_class_exists(&url, &class_name).await
            } else {
                eprintln!("Error: No CSS command specified");
                eprintln!("Usage: dx css --class-exists <CLASS> <URL>");
                ExitCode::from(1)
            }
        }
    }
}

async fn handle_class_exists(url: &str, class_name: &str) -> ExitCode {
    let cmd = CssCommand::new();
    
    match cmd.check_class_exists(url, class_name).await {
        Ok(exists) => {
            cmd.print_class_exists_result(class_name, exists);
            if exists {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(4) // Assertion failed
            }
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            match e {
                dx::error::DxError::Network(_) => ExitCode::from(2),
                dx::error::DxError::Parse(_) => ExitCode::from(3),
                dx::error::DxError::Timeout { .. } => ExitCode::from(5),
                _ => ExitCode::from(1),
            }
        }
    }
}
