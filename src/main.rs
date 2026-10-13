use clap::Parser;
use developer_assistant::{
    assistant::{answer_question, escape_terminal_field, print_result},
    cli::{Action, Cli},
    config::UserConfig,
    mcp, repl,
};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Ошибка: {}", escape_terminal_field(&error.to_string()));
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let Cli {
        project_root,
        config: config_path,
        model: model_override,
        command,
    } = Cli::parse();

    if matches!(&command, Some(Action::McpServer)) {
        return mcp::serve_stdio(&project_root).await;
    }

    let mut config = UserConfig::load(config_path.as_deref())?;
    let selected_model = model_override.or_else(|| config.default_model().map(str::to_owned));
    if let Some(model_id) = selected_model.as_deref() {
        config.resolve_model(model_id)?;
    }

    match command {
        Some(Action::Help { question }) => {
            let question = question.join(" ");
            answer_question(&project_root, &question, &config, selected_model.as_deref())
                .await
                .map(print_result)
        }
        Some(Action::McpServer) => unreachable!("MCP server path returned before config load"),
        None => repl::run(&project_root, &mut config, selected_model).await,
    }
}
