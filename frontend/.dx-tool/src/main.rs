// Created by: TEAM-DX-001
// TEAM-DX-002: Added HTML commands and JSON output format
// Frontend DX CLI Tool - Main entry point

use clap::{Parser, Subcommand, ValueEnum};
use dx::commands::{CssCommand, HtmlCommand};
use dx::config::Config;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "dx")]
#[command(version)]
#[command(about = "Frontend DX CLI Tool - Verify CSS/HTML without browser access", long_about = None)]
struct Cli {
    /// Project to target (commercial or storybook)
    #[arg(short, long, global = true)]
    project: Option<String>,
    
    /// Output format
    #[arg(short, long, global = true, value_enum, default_value = "text")]
    format: OutputFormat,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// CSS verification commands
    Css {
        /// Check if a CSS class exists in stylesheets
        #[arg(long, value_name = "CLASS")]
        class_exists: Option<String>,
        
        /// Get computed styles for a selector
        #[arg(long, value_name = "SELECTOR")]
        selector: Option<String>,
        
        /// List all classes on an element
        #[arg(long)]
        list_classes: bool,
        
        /// Selector for list-classes (required if --list-classes is set)
        #[arg(long, value_name = "SELECTOR", requires = "list_classes")]
        list_selector: Option<String>,
        
        /// URL to fetch and analyze (optional if --project is specified)
        url: Option<String>,
    },
    /// HTML structure queries
    Html {
        /// Query DOM structure with a selector
        #[arg(long, value_name = "SELECTOR")]
        selector: Option<String>,
        
        /// Get element attributes
        #[arg(long)]
        attrs: bool,
        
        /// Visualize DOM tree
        #[arg(long)]
        tree: bool,
        
        /// Maximum depth for tree visualization
        #[arg(long, default_value = "3")]
        depth: usize,
        
        /// URL to fetch and analyze (optional if --project is specified)
        url: Option<String>,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let config = Config::load();
    let use_json = matches!(cli.format, OutputFormat::Json);
    
    match cli.command {
        Commands::Css { class_exists, selector, list_classes, list_selector, url } => {
            let target_url = match resolve_url(url, cli.project.as_deref(), &config) {
                Ok(url) => url,
                Err(msg) => {
                    if use_json {
                        println!("{{\"error\": \"{}\"}}", msg.replace('"', "\\\""));
                    } else {
                        eprintln!("✗ Error: {}", msg);
                        eprintln!("\nUsage:");
                        eprintln!("  dx css --class-exists <CLASS> <URL>");
                        eprintln!("  dx --project commercial css --class-exists <CLASS>");
                    }
                    return ExitCode::from(1);
                }
            };
            
            if let Some(class_name) = class_exists {
                handle_class_exists(&target_url, &class_name, use_json).await
            } else if let Some(sel) = selector {
                handle_selector_styles(&target_url, &sel, use_json).await
            } else if list_classes {
                if let Some(sel) = list_selector {
                    handle_list_classes(&target_url, &sel, use_json).await
                } else {
                    eprintln!("Error: --list-selector required with --list-classes");
                    ExitCode::from(1)
                }
            } else {
                eprintln!("Error: No CSS command specified");
                eprintln!("Usage: dx css --class-exists <CLASS> [URL]");
                eprintln!("       dx css --selector <SELECTOR> [URL]");
                eprintln!("       dx css --list-classes --list-selector <SELECTOR> [URL]");
                ExitCode::from(1)
            }
        }
        Commands::Html { selector, attrs, tree, depth, url } => {
            let target_url = match resolve_url(url, cli.project.as_deref(), &config) {
                Ok(url) => url,
                Err(msg) => {
                    if use_json {
                        println!("{{\"error\": \"{}\"}}", msg.replace('"', "\\\""));
                    } else {
                        eprintln!("✗ Error: {}", msg);
                    }
                    return ExitCode::from(1);
                }
            };
            
            if let Some(sel) = selector {
                if tree {
                    handle_tree(&target_url, &sel, depth, use_json).await
                } else if attrs {
                    handle_attrs(&target_url, &sel, use_json).await
                } else {
                    handle_query_selector(&target_url, &sel, use_json).await
                }
            } else {
                eprintln!("Error: No HTML command specified");
                eprintln!("Usage: dx html --selector <SELECTOR> [URL]");
                eprintln!("       dx html --selector <SELECTOR> --attrs [URL]");
                eprintln!("       dx html --selector <SELECTOR> --tree [URL]");
                ExitCode::from(1)
            }
        }
    }
}

fn resolve_url(url: Option<String>, project: Option<&str>, config: &Config) -> Result<String, String> {
    // Explicit URL takes precedence
    if let Some(url) = url {
        return Ok(url);
    }
    
    // Try to resolve from project
    if let Some(project_name) = project {
        if let Some(project_config) = config.get_project(project_name) {
            return Ok(project_config.url.clone());
        } else {
            return Err(format!(
                "Unknown project '{}'. Valid projects: commercial, commercial-story, storybook\n\
                 Hint: Check .dxrc.json for available projects",
                project_name
            ));
        }
    }
    
    // No URL or project specified
    Err("No URL specified. Provide a URL or use --project <commercial|commercial-story|storybook>".to_string())
}

async fn handle_class_exists(url: &str, class_name: &str, use_json: bool) -> ExitCode {
    let cmd = CssCommand::new();
    
    match cmd.check_class_exists(url, class_name).await {
        Ok(exists) => {
            if use_json {
                println!("{{\"class\": \"{}\", \"exists\": {}}}", class_name, exists);
            } else {
                cmd.print_class_exists_result(class_name, exists);
            }
            if exists {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(4)
            }
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            match e {
                dx::error::DxError::Network(_) => ExitCode::from(2),
                dx::error::DxError::Parse(_) => ExitCode::from(3),
                dx::error::DxError::Timeout { .. } => ExitCode::from(5),
                _ => ExitCode::from(1),
            }
        }
    }
}

async fn handle_selector_styles(url: &str, selector: &str, use_json: bool) -> ExitCode {
    let cmd = CssCommand::new();
    
    match cmd.get_selector_styles(url, selector).await {
        Ok(styles) => {
            if use_json {
                match serde_json::to_string(&styles) {
                    Ok(json) => println!("{{\"selector\": \"{}\", \"styles\": {}}}", selector, json),
                    Err(_) => println!("{{\"error\": \"Failed to serialize styles\"}}"),
                }
            } else {
                cmd.print_selector_styles(selector, &styles);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            ExitCode::from(1)
        }
    }
}

async fn handle_list_classes(url: &str, selector: &str, use_json: bool) -> ExitCode {
    let cmd = CssCommand::new();
    
    match cmd.list_classes(url, selector).await {
        Ok(classes) => {
            if use_json {
                match serde_json::to_string(&classes) {
                    Ok(json) => println!("{{\"selector\": \"{}\", \"classes\": {}}}", selector, json),
                    Err(_) => println!("{{\"error\": \"Failed to serialize classes\"}}"),
                }
            } else {
                cmd.print_classes_list(selector, &classes);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            ExitCode::from(1)
        }
    }
}

async fn handle_query_selector(url: &str, selector: &str, use_json: bool) -> ExitCode {
    let cmd = HtmlCommand::new();
    
    match cmd.query_selector(url, selector).await {
        Ok(info) => {
            if use_json {
                println!("{{\"selector\": \"{}\", \"tag\": \"{}\", \"count\": {}}}", 
                    selector, info.tag, info.count);
            } else {
                cmd.print_element_info(selector, &info);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            ExitCode::from(1)
        }
    }
}

async fn handle_attrs(url: &str, selector: &str, use_json: bool) -> ExitCode {
    let cmd = HtmlCommand::new();
    
    match cmd.get_attributes(url, selector).await {
        Ok(attrs) => {
            if use_json {
                match serde_json::to_string(&attrs) {
                    Ok(json) => println!("{{\"selector\": \"{}\", \"attributes\": {}}}", selector, json),
                    Err(_) => println!("{{\"error\": \"Failed to serialize attributes\"}}"),
                }
            } else {
                cmd.print_attributes(selector, &attrs);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            ExitCode::from(1)
        }
    }
}

async fn handle_tree(url: &str, selector: &str, depth: usize, use_json: bool) -> ExitCode {
    let cmd = HtmlCommand::new();
    
    match cmd.get_tree(url, selector, depth).await {
        Ok(tree) => {
            if use_json {
                println!("{{\"selector\": \"{}\", \"tree\": \"(JSON tree not implemented)\"}}", selector);
            } else {
                cmd.print_tree(selector, &tree);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Error: {}", e);
            }
            ExitCode::from(1)
        }
    }
}
