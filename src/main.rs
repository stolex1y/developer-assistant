use clap::Parser;
use developer_assistant::{
    assistant::{answer_question, escape_terminal_field, print_result},
    cli::{Action, Cli},
    mcp,
};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Action::Help { question } => {
            let question = question.join(" ");
            answer_question(&cli.project_root, &question)
                .await
                .map(print_result)
        }
        Action::McpServer => mcp::serve_stdio(&cli.project_root).await,
    };

    if let Err(error) = result {
        let message = error.to_string();
        eprintln!("Ошибка: {}", escape_terminal_field(&message));
        std::process::exit(1);
    }
}
