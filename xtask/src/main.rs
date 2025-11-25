use anyhow::Result;
use clap::Parser;

mod cli;
mod deploy; // TEAM-451: Cloudflare deployment
mod release; // TEAM-451: Release management
mod tasks; // TEAM_527: Only rbee wrapper remains under tasks

use crate::cli::{Cmd, Xtask};

// TEAM-309: Custom narration formatter for clean output
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::registry::LookupSpan;

struct NarrationFormatter;

impl<S, N> FormatEvent<S, N> for NarrationFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        use tracing::field::{Field, Visit};

        // Extract fields from the event
        // TEAM-311: Added fn_name field
        struct FieldVisitor {
            actor: Option<String>,
            action: Option<String>,
            target: Option<String>,
            human: Option<String>,
            fn_name: Option<String>,
        }

        impl Visit for FieldVisitor {
            fn record_str(&mut self, field: &Field, value: &str) {
                match field.name() {
                    "actor" => self.actor = Some(value.to_string()),
                    "action" => self.action = Some(value.to_string()),
                    "target" => self.target = Some(value.to_string()),
                    "human" => self.human = Some(value.to_string()),
                    "fn_name" => self.fn_name = Some(value.to_string()),
                    _ => {}
                }
            }

            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                match field.name() {
                    "actor" => {
                        self.actor = Some(format!("{:?}", value).trim_matches('"').to_string())
                    }
                    "action" => {
                        self.action = Some(format!("{:?}", value).trim_matches('"').to_string())
                    }
                    "target" => {
                        self.target = Some(format!("{:?}", value).trim_matches('"').to_string())
                    }
                    "human" => {
                        self.human = Some(format!("{:?}", value).trim_matches('"').to_string())
                    }
                    "fn_name" => {
                        self.fn_name = Some(format!("{:?}", value).trim_matches('"').to_string())
                    }
                    _ => {}
                }
            }
        }

        let mut visitor =
            FieldVisitor { actor: None, action: None, target: None, human: None, fn_name: None };

        event.record(&mut visitor);

        // TEAM-310: Use centralized format_message from narration-core
        // TEAM-311: Now uses format_message_with_fn to show function names
        // TEAM-312: Removed actor from formatting - fn_name provides full trace
        // Format: Bold fn_name (40 chars), dimmed action (20 chars), message on second line
        if let (Some(action), Some(human)) = (visitor.action, visitor.human) {
            // TEAM-311: Use format_message_with_fn to include function name
            let formatted = observability_narration_core::format::format_message(
                &action,
                &human,
                visitor.fn_name.as_deref().unwrap_or("unknown"),
            );
            write!(writer, "{}", formatted)
        } else {
            // Fallback for non-narration events
            writeln!(writer, "{:?}", event)
        }
    }
}

fn main() -> Result<()> {
    // TEAM-309: Set up tracing subscriber for narration visibility
    // This makes auto-update narration visible to users
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{fmt, EnvFilter, Layer};

    let narration_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .event_format(NarrationFormatter)
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")));

    tracing_subscriber::registry().with(narration_layer).init();

    let xt = Xtask::parse();
    match xt.cmd {
        // TEAM_527: Ancient BDD/CI/regen/worker/e2e commands removed; xtask now focuses on release/deploy/rbee.
        Cmd::Release { app, r#type, dry_run, ci } => release::run(app, r#type, dry_run, ci)?,
        // TEAM-451: Cloudflare deployment
        // TEAM-463: Added production flag
        Cmd::Deploy { app, bump, production, dry_run } => {
            deploy::run(&app, bump.as_deref(), production, dry_run)?
        }
        // TEAM_531: Interactive Turbo dev scope selector
        Cmd::DevScope { print_only } => tasks::dev_scope::run(print_only)?,
        Cmd::Rbee { args } => tasks::rbee::run_rbee_keeper(args)?,
    }
    Ok(())
}
