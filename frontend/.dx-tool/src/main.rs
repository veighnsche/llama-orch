// Created by: TEAM-DX-001
// TEAM-DX-002: Added HTML commands and JSON output format
// TEAM-DX-003: Added story file locator command
// TEAM-DX-004: Added list-stories and list-variants commands
// TEAM-DX-007: Added health check command for server verification
// Frontend DX CLI Tool - Main entry point

use clap::{Parser, Subcommand, ValueEnum};
use dx::commands::{CssCommand, HtmlCommand, StoryCommand, InspectCommand, ListStoriesCommand, ListVariantsCommand};
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
    /// Story file locator
    #[command(name = "story-file")]
    StoryFile {
        /// Storybook URL to locate
        url: String,
    },
    /// Inspect element: get HTML + all related CSS in one command
    Inspect {
        /// CSS selector to inspect
        selector: String,
        
        /// URL to fetch and analyze (optional if --project is specified)
        url: Option<String>,
    },
    /// List all available stories (components) from Histoire/Storybook
    #[command(name = "list-stories")]
    ListStories {
        /// Base URL of Histoire/Storybook
        url: String,
        
        /// Filter by component name (optional)
        #[arg(long)]
        component: Option<String>,
    },
    /// List all variants for a specific story
    #[command(name = "list-variants")]
    ListVariants {
        /// Story URL to list variants for
        url: String,
        
        /// Output in copy-pastable format (just URLs with comments)
        #[arg(long)]
        copy_pastable: bool,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let config = Config::load();
    let use_json = matches!(cli.format, OutputFormat::Json);
    
    match cli.command {
        Commands::ListStories { url, component } => {
            // Preflight health check
            if let Err(code) = preflight_health_check(&url, use_json).await {
                return code;
            }
            
            let cmd = ListStoriesCommand::new();
            match cmd.list_stories(&url).await {
                Ok(stories) => {
                    if let Some(filter) = component {
                        // Filter stories by component name
                        let filtered: Vec<_> = stories.iter()
                            .filter(|s| s.component_name.to_lowercase() == filter.to_lowercase())
                            .collect();
                        
                        if filtered.is_empty() {
                            if use_json {
                                println!("{{\"error\": \"Component '{}' not found\"}}",  filter.replace('"', "\\\""));
                            } else {
                                eprintln!("✗ Component '{}' not found", filter);
                            }
                            return ExitCode::from(1);
                        }
                        
                        if use_json {
                            println!("{{\"stories\": [");
                            for (i, story) in filtered.iter().enumerate() {
                                println!("  {{\"name\": \"{}\", \"category\": \"{}\", \"url\": \"{}\"}}{}",
                                    story.component_name, story.category, story.url,
                                    if i < filtered.len() - 1 { "," } else { "" });
                            }
                            println!("]}}");
                        } else {
                            cmd.print_stories(&filtered.iter().map(|&s| s.clone()).collect::<Vec<_>>());
                        }
                    } else {
                        if use_json {
                            println!("{{\"stories\": [");
                            for (i, story) in stories.iter().enumerate() {
                                println!("  {{\"name\": \"{}\", \"category\": \"{}\", \"url\": \"{}\"}}{}",
                                    story.component_name, story.category, story.url,
                                    if i < stories.len() - 1 { "," } else { "" });
                            }
                            println!("]}}");
                        } else {
                            cmd.print_stories(&stories);
                        }
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
        Commands::ListVariants { url, copy_pastable } => {
            // Preflight health check
            if let Err(code) = preflight_health_check(&url, use_json).await {
                return code;
            }
            
            let cmd = ListVariantsCommand::new();
            match cmd.list_variants(&url).await {
                Ok(variants) => {
                    if use_json {
                        println!("{{\"variants\": [");
                        for (i, variant) in variants.iter().enumerate() {
                            println!("  {{\"id\": \"{}\", \"title\": \"{}\", \"url\": \"{}\"}}{}",
                                variant.variant_id, variant.title, variant.url,
                                if i < variants.len() - 1 { "," } else { "" });
                        }
                        println!("]}}");
                    } else if copy_pastable {
                        cmd.print_variants_copy_pastable(&variants);
                    } else {
                        cmd.print_variants(&variants, &url);
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
        Commands::Inspect { selector, url } => {
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
            
            // Preflight health check
            if let Err(code) = preflight_health_check(&target_url, use_json).await {
                return code;
            }
            
            handle_inspect(&target_url, &selector, use_json).await
        }
        Commands::StoryFile { url } => {
            let cmd = StoryCommand::new();
            match cmd.locate_story_file(&url) {
                Ok(info) => {
                    if use_json {
                        let component_file_str = info.component_file
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "null".to_string());
                        println!("{{\"story_file\": \"{}\", \"component_file\": \"{}\", \"story_path\": \"{}\"}}",
                            info.story_file.display(), component_file_str, info.story_path);
                    } else {
                        cmd.print_story_info(&info);
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
            
            // Preflight health check
            if let Err(code) = preflight_health_check(&target_url, use_json).await {
                return code;
            }
            
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
            
            // Preflight health check
            if let Err(code) = preflight_health_check(&target_url, use_json).await {
                return code;
            }
            
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

async fn handle_inspect(url: &str, selector: &str, use_json: bool) -> ExitCode {
    let cmd = InspectCommand::new();
    
    match cmd.inspect_element(url, selector).await {
        Ok(result) => {
            if use_json {
                // Build JSON output
                let classes_json = serde_json::to_string(&result.classes).unwrap_or_else(|_| "[]".to_string());
                let attrs_json = serde_json::to_string(&result.attributes).unwrap_or_else(|_| "{}".to_string());
                println!("{{\"selector\": \"{}\", \"tag\": \"{}\", \"classes\": {}, \"attributes\": {}, \"element_count\": {}}}",
                    selector, result.tag, classes_json, attrs_json, result.element_count);
            } else {
                cmd.print_inspect_result(&result);
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

// TEAM-DX-007: Preflight health check - runs before every command that needs a server
// This prevents cryptic errors and gives clear feedback when server is down
async fn preflight_health_check(url: &str, use_json: bool) -> Result<(), ExitCode> {
    use std::time::Duration;
    use reqwest::Client;
    
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    
    match client.get(url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(())
            } else {
                let status = response.status().as_u16();
                if use_json {
                    println!("{{\"error\": \"Server returned status {}\"}}", status);
                } else {
                    eprintln!("✗ Server returned error status: {}", status);
                    eprintln!("  URL: {}", url);
                }
                Err(ExitCode::from(3))
            }
        }
        Err(e) => {
            if use_json {
                println!("{{\"error\": \"Server not responding: {}\"}}", 
                    e.to_string().replace('"', "\\\""));
            } else {
                eprintln!("✗ Server is not responding");
                eprintln!("  URL: {}", url);
                eprintln!("  Error: {}", e);
                eprintln!("\nPossible causes:");
                eprintln!("  - Server not started");
                eprintln!("  - Wrong port number (check URL)");
                eprintln!("  - Server crashed");
                eprintln!("\nSuggestions:");
                eprintln!("  - Start Histoire: cd frontend/bin/commercial && pnpm run story:dev");
                eprintln!("  - Or use correct port: --project commercial-story (port 6007)");
                eprintln!("  - Check if process is running: ps aux | grep histoire");
            }
            Err(ExitCode::from(2))
        }
    }
}
