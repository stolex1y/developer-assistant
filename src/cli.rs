use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "developer-assistant",
    version,
    about = "Ассистент разработчика: read-only /help и изменения по цели /task с подтверждением diff",
    disable_help_subcommand = true
)]
pub struct Cli {
    #[arg(long, global = true, default_value = ".")]
    pub project_root: PathBuf,

    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,

    #[arg(long, global = true, value_name = "PROVIDER/MODEL")]
    pub model: Option<String>,

    #[command(subcommand)]
    pub command: Option<Action>,
}

#[derive(Debug, Subcommand)]
pub enum Action {
    #[command(name = "/help")]
    Help {
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        question: Vec<String>,
    },
    #[command(
        name = "/task",
        about = "Предложить изменения по цели; применить только после подтверждения diff"
    )]
    Task {
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        goal: Vec<String>,
    },
    #[command(name = "__mcp-server", hide = true)]
    McpServer,
    #[command(name = "review", about = "Выполнить read-only AI review pull request")]
    Review {
        #[arg(long, value_name = "OWNER/REPO")]
        repository: String,
        #[arg(long, value_name = "NUMBER")]
        pr_number: u64,
        #[arg(long, value_name = "BASE_SHA")]
        base_sha: String,
        #[arg(long, value_name = "HEAD_SHA")]
        head_sha: String,
        #[arg(long, value_name = "TOKENS")]
        max_uncached_input_tokens: u64,
    },
}
