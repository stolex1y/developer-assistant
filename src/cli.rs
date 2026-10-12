use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "developer-assistant",
    version,
    about = "Ассистент проекта с доступом только для чтения: ответы по README и docs",
    disable_help_subcommand = true
)]
pub struct Cli {
    #[arg(long, global = true, default_value = ".")]
    pub project_root: PathBuf,

    #[command(subcommand)]
    pub command: Action,
}

#[derive(Debug, Subcommand)]
pub enum Action {
    #[command(name = "/help")]
    Help {
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        question: Vec<String>,
    },
    #[command(name = "__mcp-server", hide = true)]
    McpServer,
}
