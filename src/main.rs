use anyhow::{Context, anyhow, bail};
use clap::Parser;
use developer_assistant::{
    assistant::{answer_question, escape_terminal_field, print_result},
    cli::{Action, Cli},
    config::UserConfig,
    mcp, repl,
    retrieval::ProjectIndex,
    review::{
        deepseek::DeepSeekClient,
        embedding::FastEmbedder,
        github::{GitHubClient, PullRequestId},
        index::ReviewIndex,
        runner::{ReviewRunner, validate_checkout},
    },
};
use reqwest::Url;
use std::{env, path::Path};

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
            let mut project_index = ProjectIndex::default();
            answer_question(
                &project_root,
                &question,
                &config,
                selected_model.as_deref(),
                &mut project_index,
            )
            .await
            .map(print_result)
        }
        Some(Action::Review {
            repository,
            pr_number,
            base_sha,
            head_sha,
            max_uncached_input_tokens,
        }) => {
            run_review(
                &project_root,
                &config,
                selected_model.as_deref(),
                ReviewInvocation {
                    repository,
                    pr_number,
                    base_sha,
                    head_sha,
                    max_uncached_input_tokens,
                },
            )
            .await
        }
        Some(Action::McpServer) => unreachable!("MCP server path returned before config load"),
        None => repl::run(&project_root, &mut config, selected_model).await,
    }
}

struct ReviewInvocation {
    repository: String,
    pr_number: u64,
    base_sha: String,
    head_sha: String,
    max_uncached_input_tokens: u64,
}

async fn run_review(
    project_root: &Path,
    config: &UserConfig,
    selected_model: Option<&str>,
    invocation: ReviewInvocation,
) -> anyhow::Result<()> {
    let ReviewInvocation {
        repository,
        pr_number,
        base_sha,
        head_sha,
        max_uncached_input_tokens,
    } = invocation;
    if max_uncached_input_tokens == 0 {
        bail!("Лимит uncached input tokens должен быть больше нуля.");
    }
    let selected_model =
        selected_model.context("Для review задайте --model или DeepSeek default в config.")?;
    let selection = config.resolve_model(selected_model)?;
    if selection.provider_id() != "deepseek" {
        bail!("AI code review в этой версии использует только provider deepseek.");
    }
    let provider = selection.provider_config()?;
    let deepseek = DeepSeekClient::new(
        provider.endpoint().clone(),
        provider.api_key().to_owned(),
        provider.model().to_owned(),
    )?;

    let github_token = env::var("GITHUB_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("Задайте GitHub Actions token в GITHUB_TOKEN."))?;
    let github_api = Url::parse("https://api.github.com/")?;
    let github = GitHubClient::new(github_api, github_token)?;

    let (owner, repository_name) = repository
        .split_once('/')
        .context("--repository должен иметь формат OWNER/REPO.")?;
    let identity = PullRequestId {
        owner: owner.to_owned(),
        repository: repository_name.to_owned(),
        number: pr_number,
        expected_base_sha: base_sha,
        expected_head_sha: head_sha,
    };
    identity.validate()?;
    validate_checkout(project_root, &identity.expected_head_sha)?;

    let cache_root = project_root.join(".data/developer-assistant-review");
    let embedder = FastEmbedder::load(project_root, &cache_root.join("model"))?;
    let mut index = ReviewIndex::open(project_root, &cache_root.join("index"), embedder)?;
    let runner = ReviewRunner::new(github, deepseek);
    let result = runner
        .run_with_index(
            project_root,
            &identity,
            max_uncached_input_tokens,
            &mut index,
        )
        .await?;

    println!(
        "Review PR #{pr_number}: files={}, findings={}, threads updated={}, threads resolved={}.",
        result.changed_files_reviewed,
        result.findings_published,
        result.threads_created_or_updated,
        result.threads_resolved
    );
    println!(
        "Budgeted uncached input tokens (retrieval + focused assessment): {}; tool calls: {}; provider requests: {}.",
        result.retrieval_uncached_input_tokens, result.tool_calls_executed, result.requests
    );
    if let Some(usage) = result.final_usage {
        println!(
            "Final no-tools request usage (separate): prompt={}, cached={}, uncached={}.",
            usage.prompt_tokens, usage.cached_input_tokens, usage.uncached_input_tokens
        );
    }
    Ok(())
}
