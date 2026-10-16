use serde_json::{Value, json};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output, Stdio},
    time::Duration,
};
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    process::Command,
    task::JoinHandle,
    time::timeout,
};

const TEST_API_KEY: &str = "local-test-key-not-a-credential";
const STRUCTURE_QUESTION: &str =
    "Какие Gradle-модули есть, где web entry point и что описывает schema transaction-import.v3?";
const STRUCTURE_ANSWER: &str = "Проект делится на core, transport, persistence и web; основной интерфейс — GET /agent; schema transaction-import.v3 описывает импорт операций.";
const MARKDOWN_ANSWER: &str = r#"# D31 preview

**Bold** and *italic*.

- unordered item
1. ordered item

Inline `component()`.

```rust
fn render() {}
```

[Project docs](https://example.test/docs)

| Item | State |
| --- | --- |
| renderer | ready |"#;
struct CapturedRequest {
    request_line: String,
    body: Vec<u8>,
    authorization: Option<String>,
}

#[tokio::test]
async fn help_uses_docs_and_live_mcp_context() {
    let (_fixture, root) = selected_project_or_fixture();
    let expected_branch = git_output(&root, &["branch", "--show-current"]);
    let before = git_output(&root, &["status", "--porcelain", "--untracked-files=all"]);
    let responses = vec![
        tool_call_response(
            "schema-search",
            "search_code",
            json!({"query": "transaction-import.v3 schema", "limit": 8}),
            12,
        ),
        completion_response(STRUCTURE_ANSWER),
    ];
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        &root,
        STRUCTURE_QUESTION,
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not complete the provider tool loop")
        .expect("fake provider task failed");
    let after = git_output(&root, &["status", "--porcelain", "--untracked-files=all"]);

    assert_eq!(
        before, after,
        "CLI changed the selected project working tree"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let normalized = stdout.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(normalized.contains("core"));
    assert!(normalized.contains("GET /agent"));
    assert!(normalized.contains("transaction-import.v3"));
    assert!(stdout.contains("docs/architecture.md"));
    assert!(stdout.contains("Источники:"));

    assert_eq!(requests.len(), 2);
    let request: Value =
        serde_json::from_slice(&requests[0].body).expect("fake provider got valid JSON");
    let followup: Value =
        serde_json::from_slice(&requests[1].body).expect("tool follow-up is valid JSON");
    let search_result = followup["messages"]
        .as_array()
        .expect("follow-up messages exist")
        .iter()
        .find(|message| message["role"] == "tool")
        .and_then(|message| message["content"].as_str())
        .expect("search_code result is sent back to the model");
    assert!(search_result.contains("docs/data-api.md"));
    assert!(search_result.contains("transaction-import.v3"));
    assert!(String::from_utf8_lossy(&output.stdout).contains("docs/data-api.md#"));

    assert_eq!(request["model"], "gpt-4o-mini");
    assert_eq!(request["temperature"], 0.0);
    assert_eq!(request["max_tokens"], 800);
    assert_eq!(request["tool_choice"], "auto");
    assert_eq!(request["tools"].as_array().unwrap().len(), 2);
    assert_eq!(request["tools"][0]["function"]["name"], "search_code");
    assert_eq!(request["tools"][1]["function"]["name"], "read_file");
    let messages = request["messages"].as_array().expect("chat messages array");
    let prompt = messages
        .iter()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .expect("user prompt exists");
    assert!(prompt.contains(&format!("Текущая Git-ветка: {expected_branch}")));
    assert!(prompt.contains("Пути проекта ("));
    assert!(prompt.contains("README.md"));
    assert!(prompt.contains("[docs/architecture.md#"));
    assert!(prompt.contains("core"));
    assert!(!prompt.contains(TEST_API_KEY));
}

#[tokio::test]
async fn help_renders_markdown_in_one_shot_cli() {
    let fixture = git_project_with_docs(
        "# Overview\nThe read-only project interface is available for structure questions.",
        "# Architecture\nThe service exposes a local interface.",
    );
    let (base_url, fake_provider) =
        start_provider("200 OK", completion_response(MARKDOWN_ANSWER)).await;

    let output = run_cli(
        fixture.path(),
        "Что описывает интерфейс проекта?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let _request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("one-shot CLI did not call fake provider")
        .expect("fake provider task completed");

    assert_readable_markdown_answer(&String::from_utf8_lossy(&output.stdout));
}

#[tokio::test]
async fn help_renders_markdown_in_repl() {
    let fixture = git_project_with_docs(
        "# Overview\nThe read-only project interface is available for structure questions.",
        "# Architecture\nThe service exposes a local interface.",
    );
    let (base_url, fake_provider) =
        start_provider("200 OK", completion_response(MARKDOWN_ANSWER)).await;
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    fs::write(
        &config_path,
        format!(
            "[providers.test]\nbase_url = \"{base_url}\"\napi_key_env = \"DEVELOPER_ASSISTANT_API_KEY\"\nmodels = [\"test-model\"]\n\n[selection]\ndefault_model = \"test/test-model\"\n"
        ),
    )
    .expect("provider catalog is written");

    let output = run_repl(
        fixture.path(),
        &config_path,
        None,
        "/help Что описывает интерфейс проекта?\n/exit\n",
        &[("DEVELOPER_ASSISTANT_API_KEY", TEST_API_KEY)],
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("REPL failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let _request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("REPL did not call fake provider")
        .expect("fake provider task completed");

    assert_readable_markdown_answer(&String::from_utf8_lossy(&output.stdout));
}

fn assert_readable_markdown_answer(stdout: &str) {
    let answer = stdout
        .split_once("Ответ:\n")
        .expect("answer section is printed")
        .1
        .split_once("\n\nToken budget:")
        .expect("token diagnostics follow the answer")
        .0;
    for visible in [
        "D31 preview",
        "Bold",
        "italic",
        "unordered item",
        "ordered item",
        "component()",
        "fn render() {}",
        "Project docs",
        "https://example.test/docs",
        "renderer",
        "ready",
    ] {
        assert!(
            answer.contains(visible),
            "missing rendered content: {visible}"
        );
    }
    assert!(!answer.contains("##"));
    assert!(!answer.contains("**"));
    assert!(!answer.contains("```"));
    assert!(!answer.contains("| Item |"));
    assert!(!answer.contains("[Project docs]"));
    assert!(!stdout.contains('\u{1b}'));
    assert!(
        stdout
            .split_once("\n\nToken budget:")
            .expect("diagnostics follow answer")
            .1
            .contains("Источники:")
    );
}

#[tokio::test]
async fn help_retrieves_readme_as_a_cited_source() {
    let (_fixture, root) = selected_project_or_fixture();
    let response = completion_response("README описывает read-only trace проекта.");
    let (base_url, fake_provider) = start_provider("200 OK", response).await;

    let output = run_cli(
        &root,
        "Опишите read-only trace в README.",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");
    let request: Value = serde_json::from_slice(&request).expect("fake provider got valid JSON");
    let prompt = user_prompt(&request);

    assert!(prompt.contains("[README.md#"));
    assert!(prompt.contains("read-only trace"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("README.md#"));
    assert!(stdout.contains("Источники:"));
}

#[tokio::test]
async fn help_read_file_returns_heading_and_prints_citable_markdown_source() {
    let fixture = git_project_with_docs(
        "# Overview\nThe repository has a small local interface.",
        "# Architecture\nThe service exposes a local interface.",
    );
    let responses = vec![
        tool_call_response(
            "read-architecture",
            "read_file",
            json!({"path": "docs/architecture.md", "start_line": 2, "end_line": 2}),
            12,
        ),
        completion_response("Интерфейс описан в архитектурной документации."),
    ];
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        fixture.path(),
        "Как устроен интерфейс?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("tool loop did not complete")
        .expect("fake provider task completed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 2);
    let followup: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_result = followup["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .and_then(|message| message["content"].as_str())
        .expect("read_file result is sent back to the model");
    let payload: Value = serde_json::from_str(tool_result).unwrap();
    assert_eq!(payload["heading"], "Architecture");
    assert!(
        payload["text"]
            .as_str()
            .unwrap()
            .contains("The service exposes a local interface.")
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("`docs/architecture.md#Architecture` (строки 2–2)")
    );
}

#[tokio::test]
async fn help_preserves_large_quoted_read_file_result() {
    let quoted = "\"".repeat(6_000);
    let docs = format!("# Architecture\n{quoted}");
    let fixture = git_project_with_docs("# Overview\nThe repository has a local interface.", &docs);
    let responses = vec![
        tool_call_response(
            "read-quoted",
            "read_file",
            json!({"path": "docs/architecture.md", "start_line": 2, "end_line": 2}),
            12,
        ),
        completion_response("Цитируемый фрагмент получен."),
    ];
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        fixture.path(),
        "Покажи документированный пример.",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("tool loop did not complete")
        .expect("fake provider task completed");

    assert!(output.status.success());
    let followup: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_result = followup["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .and_then(|message| message["content"].as_str())
        .expect("read_file result is sent back to the model");
    assert!(tool_result.len() > 12_000);
    let payload: Value = serde_json::from_str(tool_result).unwrap();
    assert_eq!(payload["path"], "docs/architecture.md");
    assert_eq!(payload["heading"], "Architecture", "{payload}");
    assert_eq!(payload["start_line"], 2);
    assert_eq!(payload["end_line"], 2);
    assert!(payload["text"].as_str().unwrap().contains(&quoted));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("`docs/architecture.md#Architecture` (строки 2–2)"),
        "{stdout}"
    );
}

#[tokio::test]
async fn help_cites_only_the_lines_returned_by_read_file() {
    let fixture = git_project_with_docs(
        "# Overview\nThe repository has a local interface.",
        "# Architecture\nThe service exposes a local interface.\n",
    );
    let responses = vec![
        tool_call_response(
            "read-architecture",
            "read_file",
            json!({"path": "docs/architecture.md", "start_line": 1, "end_line": 80}),
            12,
        ),
        completion_response("Архитектура описывает интерфейс."),
    ];
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        fixture.path(),
        "Как устроен интерфейс?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("tool loop did not complete")
        .expect("fake provider task completed");

    assert!(output.status.success());
    let followup: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_result = followup["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .and_then(|message| message["content"].as_str())
        .expect("read_file result is sent back to the model");
    let payload: Value = serde_json::from_str(tool_result).unwrap();
    assert_eq!(payload["start_line"], 1);
    assert_eq!(payload["end_line"], 2);
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("`docs/architecture.md#Architecture` (строки 1–2)")
    );
}

#[tokio::test]
async fn help_stops_tools_at_the_uncached_token_budget() {
    let fixture = git_project_with_docs(
        "# Configuration\nThe configured storage engine uses MEMORY_ENGINE_MARKER.",
        "# Architecture\nThe project exposes a small local interface.",
    );
    let responses = vec![
        tool_call_response(
            "search-call",
            "search_code",
            json!({"query": "storage engine", "limit": 3}),
            18_000,
        ),
        tool_call_response(
            "read-call",
            "read_file",
            json!({"path": "README.md", "start_line": 1, "end_line": 2}),
            12_001,
        ),
        completion_response("Лимит достигнут; возвращаю подтверждённые сведения."),
    ];
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        fixture.path(),
        "Какой storage engine указан?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("tool loop did not complete")
        .expect("fake provider task completed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 3);
    let parsed = requests
        .iter()
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(parsed[0]["tool_choice"], "auto");
    assert_eq!(parsed[1]["tool_choice"], "auto");
    assert_eq!(parsed[2]["tool_choice"], "none");
    assert!(parsed[2].get("tools").is_none());
    let final_tool_results = parsed[2]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect::<Vec<_>>();
    assert_eq!(final_tool_results.len(), 2);
    assert!(
        final_tool_results[0]["content"]
            .as_str()
            .unwrap()
            .contains("MEMORY_ENGINE_MARKER")
    );
    assert!(
        final_tool_results[1]["content"]
            .as_str()
            .unwrap()
            .contains("input_budget_exhausted")
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("30001 uncached input tokens / 30000"));
    assert!(stdout.contains("10 uncached input tokens; он не входит"));
    assert!(stdout.contains("tools executed: 1"));
}

#[tokio::test]
async fn help_has_no_fixed_tool_call_count_limit() {
    let fixture = git_project_with_docs(
        "# Configuration\nThe configured storage engine uses MEMORY_ENGINE_MARKER.",
        "# Architecture\nThe project exposes a small local interface.",
    );
    let mut responses = (0..31)
        .map(|index| {
            tool_call_response(
                &format!("search-{index}"),
                "search_code",
                json!({"query": format!("storage engine marker {index}"), "limit": 1}),
                1,
            )
        })
        .collect::<Vec<_>>();
    responses.push(completion_response(
        "Поиск завершён после нескольких уточнений.",
    ));
    let (base_url, fake_provider) = start_provider_script(responses).await;

    let output = run_cli(
        fixture.path(),
        "Где описан storage engine?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("tool loop did not complete 31 distinct tool calls")
        .expect("fake provider task completed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 32);
    let final_request: Value =
        serde_json::from_slice(&requests[31].body).expect("final request is JSON");
    assert_eq!(final_request["tool_choice"], "auto");
    assert_eq!(final_request["tools"].as_array().unwrap().len(), 2);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("41 uncached input tokens / 30000"));
    assert!(stdout.contains("provider requests: 32, tools executed: 31"));
}

#[tokio::test]
async fn help_refreshes_the_project_index_between_repl_questions() {
    let fixture = git_project_with_docs(
        "# Configuration\nThe configured storage engine uses OLD_ENGINE_MARKER.",
        "# Architecture\nThe project exposes a local interface.",
    );
    let responses = vec![
        completion_response("В первой версии использовался старый движок."),
        completion_response("После обновления используется новый движок."),
    ];
    let (base_url, fake_provider) =
        start_provider_and_update_readme(fixture.path().to_owned(), responses).await;
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    fs::write(
        &config_path,
        format!(
            "[providers.test]\nbase_url = \"{base_url}\"\napi_key_env = \"DEVELOPER_ASSISTANT_API_KEY\"\nmodels = [\"test-model\"]\n\n[selection]\ndefault_model = \"test/test-model\"\n"
        ),
    )
    .expect("provider catalog is written");

    let output = run_repl(
        fixture.path(),
        &config_path,
        None,
        "/help Какой storage engine настроен?\n/help Какой storage engine настроен?\n/exit\n",
        &[("DEVELOPER_ASSISTANT_API_KEY", TEST_API_KEY)],
    )
    .await;
    let requests = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("REPL did not complete both help requests")
        .expect("fake provider task completed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 2);
    let first_request: Value =
        serde_json::from_slice(&requests[0].body).expect("first request is JSON");
    let second_request: Value =
        serde_json::from_slice(&requests[1].body).expect("second request is JSON");
    let first_prompt = user_prompt(&first_request);
    let second_prompt = user_prompt(&second_request);
    assert!(first_prompt.contains("OLD_ENGINE_MARKER"));
    assert!(!first_prompt.contains("NEW_ENGINE_MARKER"));
    assert!(second_prompt.contains("NEW_ENGINE_MARKER"));
    assert!(!second_prompt.contains("OLD_ENGINE_MARKER"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("старый движок"));
    assert!(stdout.contains("новый движок"));
}

#[tokio::test]
async fn help_ignores_files_present_only_in_an_inherited_git_index() {
    let fixture = git_project_with_docs(
        "# Overview\nThe project has a local interface.",
        "# Architecture\nThe service exposes a local interface.",
    );
    fs::write(
        fixture.path().join("notes.md"),
        "# Notes\nUNTRACKED_SENTINEL_48291\n",
    )
    .unwrap();
    let alternate_index_directory = TempDir::new().expect("alternate Git index directory");
    let alternate_index = alternate_index_directory.path().join("alternate.index");
    let staged = StdCommand::new("git")
        .arg("-C")
        .arg(fixture.path())
        .args(["add", "--", "notes.md"])
        .env("GIT_INDEX_FILE", &alternate_index)
        .output()
        .expect("git is installed");
    assert!(
        staged.status.success(),
        "{}",
        String::from_utf8_lossy(&staged.stderr)
    );
    let (base_url, fake_provider) = start_provider(
        "200 OK",
        completion_response("Нет подтверждения в корпусе."),
    )
    .await;

    let output = run_cli_with_git_index(
        fixture.path(),
        "Что означает число 48291?",
        &base_url,
        TEST_API_KEY,
        &alternate_index,
    )
    .await;
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call the fake provider")
        .expect("fake provider task completed");
    let request: Value = serde_json::from_slice(&request).expect("provider request is valid JSON");
    let prompt = user_prompt(&request);

    assert!(output.status.success());
    assert!(!prompt.contains("UNTRACKED_SENTINEL_48291"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("notes.md"));
}

#[tokio::test]
async fn help_without_relevant_docs_skips_provider() {
    let fixture = empty_git_project();
    let impossible_endpoint = "http://127.0.0.1:1/v1";
    let output = run_cli(
        fixture.path(),
        "В каком хранилище работает проект?",
        Some(impossible_endpoint),
        Some(TEST_API_KEY),
    )
    .await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Ветка Git: `main`"));
    assert!(stdout.contains("В filtered corpus нет допустимых tracked-файлов"));
    assert!(stdout.contains("provider не вызывался"));
}

#[tokio::test]
async fn help_calls_provider_with_tools_when_initial_retrieval_has_no_match() {
    let fixture = git_project_with_docs(
        "# Overview\nThe project exposes a documented local interface.",
        "# Architecture\nThe repository contains a small service.",
    );
    let (base_url, fake_provider) = start_provider(
        "200 OK",
        completion_response("В доступном корпусе нет подтверждения."),
    )
    .await;

    let output = run_cli(
        fixture.path(),
        "Расскажи про qzxv-nonexistent-token.",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("provider was not called for a nonempty corpus")
        .expect("fake provider task completed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let request: Value = serde_json::from_slice(&request).expect("provider request is valid JSON");
    assert_eq!(request["tool_choice"], "auto");
    assert_eq!(request["tools"].as_array().unwrap().len(), 2);
    assert!(user_prompt(&request).contains("Подходящих начальных фрагментов нет"));
}

#[tokio::test]
async fn mcp_limits_file_list_and_reports_total_count() {
    let fixture = empty_git_project();
    for index in 0..55 {
        fs::write(fixture.path().join(format!("source-{index:02}.txt")), "").unwrap();
    }
    git_output(fixture.path(), &["add", "."]);

    let output = run_cli(fixture.path(), "Опиши структуру файлов.", None, None).await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Файлы через MCP: 50 из 55"));
    assert!(stdout.contains("- source-00.txt"));
    assert!(stdout.contains("- source-49.txt"));
    assert!(!stdout.contains("- source-50.txt"));
    assert!(stdout.contains("список сокращён до 50 путей"));
}

#[tokio::test]
async fn mcp_file_list_excludes_untracked_and_ignored_files() {
    let fixture = git_project_with_docs(
        "# Structure\nThe web application exposes a local route.",
        "# Entry point\nThe primary web route is /agent.",
    );
    fs::write(fixture.path().join(".gitignore"), "ignored-only.txt\n").unwrap();
    fs::write(fixture.path().join("tracked-only.txt"), "").unwrap();
    git_output(
        fixture.path(),
        &["add", ".gitignore", "README.md", "docs", "tracked-only.txt"],
    );
    fs::write(fixture.path().join("untracked-only.txt"), "").unwrap();
    fs::write(fixture.path().join("ignored-only.txt"), "").unwrap();
    let (base_url, fake_provider) = start_provider(
        "200 OK",
        completion_response("Структура описана в документации."),
    )
    .await;

    let output = run_cli(
        fixture.path(),
        "Где web entry point?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");
    let request: Value = serde_json::from_slice(&request).unwrap();
    let prompt = user_prompt(&request);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Файлы через MCP: 4 из 4"));
    assert!(stdout.contains("- tracked-only.txt"));
    assert!(!stdout.contains("untracked-only.txt"));
    assert!(!stdout.contains("ignored-only.txt"));
    assert!(prompt.contains("tracked-only.txt"));
    assert!(!prompt.contains("untracked-only.txt"));
    assert!(!prompt.contains("ignored-only.txt"));
}

#[tokio::test]
async fn mcp_file_inventory_deduplicates_unmerged_stages() {
    let fixture = empty_git_project();
    git_output(fixture.path(), &["config", "user.name", "D31 Test"]);
    git_output(
        fixture.path(),
        &["config", "user.email", "d31-test@example.invalid"],
    );
    for index in 0..48 {
        fs::write(fixture.path().join(format!("a-{index:02}.txt")), [0]).unwrap();
    }
    fs::write(fixture.path().join("conflict.txt"), [0]).unwrap();
    fs::write(fixture.path().join("z-last.txt"), [0]).unwrap();
    git_output(fixture.path(), &["add", "."]);
    git_output(fixture.path(), &["commit", "-m", "baseline"]);

    git_output(fixture.path(), &["switch", "-c", "other"]);
    fs::write(fixture.path().join("conflict.txt"), [0, 1]).unwrap();
    git_output(fixture.path(), &["add", "conflict.txt"]);
    git_output(fixture.path(), &["commit", "-m", "other conflict change"]);

    git_output(fixture.path(), &["switch", "main"]);
    fs::write(fixture.path().join("conflict.txt"), [0, 2]).unwrap();
    git_output(fixture.path(), &["add", "conflict.txt"]);
    git_output(fixture.path(), &["commit", "-m", "main conflict change"]);
    let merge = StdCommand::new("git")
        .arg("-C")
        .arg(fixture.path())
        .args(["merge", "other"])
        .output()
        .expect("git is installed");
    assert!(
        !merge.status.success(),
        "fixture must have an unresolved conflict"
    );

    let output = run_cli(fixture.path(), "Опиши структуру файлов.", None, None).await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Файлы через MCP: 50 из 50"));
    assert_eq!(stdout.matches("- conflict.txt").count(), 1);
    assert!(stdout.contains("- z-last.txt"));
}

#[cfg(unix)]
#[tokio::test]
async fn mcp_paths_escape_terminal_controls_before_output() {
    let fixture = empty_git_project();
    let path = "control\u{1b}[2J.txt";
    fs::write(fixture.path().join(path), "").unwrap();
    git_output(fixture.path(), &["add", path]);
    let output = run_cli(fixture.path(), "Опиши структуру файлов.", None, None).await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("control\\u{1b}[2J.txt"));
    assert!(!stdout.contains('\u{1b}'));
}

#[tokio::test]
async fn mcp_bidi_controls_are_escaped_before_output() {
    let fixture = empty_git_project();
    let branch = "main\u{202e}hidden";
    git_output(fixture.path(), &["branch", "-m", branch]);
    let path = "source\u{202e}.txt";
    fs::write(fixture.path().join(path), "").unwrap();
    git_output(fixture.path(), &["add", path]);

    let output = run_cli(fixture.path(), "Опиши структуру файлов.", None, None).await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Ветка Git: `main\\u{202e}hidden`"));
    assert!(stdout.contains("- source\\u{202e}.txt"));
    assert!(!stdout.contains('\u{202e}'));
}

#[tokio::test]
async fn credential_values_are_redacted_before_the_provider_request() {
    let fixture = git_project_with_docs(
        "# Credentials\nExample password: \"synthetic redaction marker\"",
        "# Overview\nThis project has a local command-line interface.",
    );
    let response = completion_response("В документации нет безопасного значения.");
    let (base_url, fake_provider) = start_provider("200 OK", response).await;

    let output = run_cli(
        fixture.path(),
        "Какое password значение приведено в документации?",
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");
    let request: Value = serde_json::from_slice(&request).unwrap();
    let prompt = user_prompt(&request);
    assert!(prompt.contains("password [REDACTED]"));
    assert!(!prompt.contains("synthetic redaction marker"));
}

#[tokio::test]
async fn provider_terminal_controls_are_escaped_before_output() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let response = completion_response("Первая строка.\n\u{1b}[2J\u{7}\u{202e}готово");
    let (base_url, fake_provider) = start_provider("200 OK", response).await;
    let output = run_cli(
        fixture.path(),
        STRUCTURE_QUESTION,
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    if !output.status.success() {
        fake_provider.abort();
        panic!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let _request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Первая строка."));
    assert!(stdout.contains("готово"));
    assert!(!stdout.contains('\u{1b}'));
    assert!(!stdout.contains('\u{7}'));
    assert!(!stdout.contains('\u{202e}'));
}

#[tokio::test]
async fn missing_api_key_fails_before_provider_request() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let output = run_cli(
        fixture.path(),
        STRUCTURE_QUESTION,
        Some("http://127.0.0.1:1/v1"),
        None,
    )
    .await;

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("DEVELOPER_ASSISTANT_API_KEY"));
    assert!(!stderr.contains("Не удалось подключиться к provider"));
    assert!(!stderr.contains(TEST_API_KEY));
}

#[tokio::test]
async fn provider_error_does_not_leak_response_body_or_api_key() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let (base_url, fake_provider) = start_provider(
        "500 Internal Server Error",
        "provider-error-marker".to_owned(),
    )
    .await;
    let output = run_cli(
        fixture.path(),
        STRUCTURE_QUESTION,
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let _request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Provider вернул HTTP 500"));
    assert!(!stderr.contains("provider-error-marker"));
    assert!(!stderr.contains(TEST_API_KEY));
}

#[tokio::test]
async fn provider_rejects_malformed_and_empty_completions() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let responses = [
        (
            "malformed-provider-body-marker".to_owned(),
            "некорректный JSON",
            Some("malformed-provider-body-marker"),
        ),
        (
            json!({
                "choices": [{"message": {"role": "assistant", "content": "  "}}],
                "usage": {
                    "prompt_tokens": 10,
                    "prompt_tokens_details": {
                        "prompt_cache_hit_tokens": 0,
                        "prompt_cache_miss_tokens": 10
                    }
                }
            })
            .to_string(),
            "пустой ответ",
            None,
        ),
        (
            json!({"choices": [{"message": {"role": "assistant", "content": "missing usage"}}]})
                .to_string(),
            "token usage",
            None,
        ),
    ];

    for (response_body, expected_error, response_marker) in responses {
        let (base_url, fake_provider) = start_provider("200 OK", response_body).await;
        let output = run_cli(
            fixture.path(),
            STRUCTURE_QUESTION,
            Some(&base_url),
            Some(TEST_API_KEY),
        )
        .await;
        let _request = timeout(Duration::from_secs(5), fake_provider)
            .await
            .expect("CLI did not call fake provider")
            .expect("fake provider task failed");

        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expected_error), "{stderr}");
        if let Some(marker) = response_marker {
            assert!(!stderr.contains(marker));
        }
    }
}

#[tokio::test]
async fn provider_does_not_follow_redirects() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let (base_url, fake_provider, redirect_target) = start_redirect_provider().await;
    let target_abort = redirect_target.abort_handle();
    let output = run_cli(
        fixture.path(),
        STRUCTURE_QUESTION,
        Some(&base_url),
        Some(TEST_API_KEY),
    )
    .await;
    let _request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call fake provider")
        .expect("fake provider task failed");
    let redirect_was_followed = match timeout(Duration::from_millis(250), redirect_target).await {
        Ok(Ok(())) => true,
        Ok(Err(error)) => panic!("redirect target failed: {error}"),
        Err(_) => {
            target_abort.abort();
            false
        }
    };

    assert!(
        !redirect_was_followed,
        "provider redirect must not be followed"
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Provider вернул HTTP 302"));
}

#[tokio::test]
async fn repl_without_config_starts_empty_and_does_not_create_a_file() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("missing.toml");
    let output = run_repl(
        fixture.path(),
        &config_path,
        None,
        &format!("/models\n/help {STRUCTURE_QUESTION}\n/exit\n"),
        &[],
    )
    .await;

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("Каталог моделей пуст"));
    assert!(stdout.contains("config.example.toml"));
    assert!(stdout.contains(&config_path.display().to_string()));
    assert!(stderr.contains("Не выбрана модель"));
    assert!(!stdout.contains("Ответ:"));
    assert!(!config_path.exists());
}

#[tokio::test]
async fn repl_selects_models_across_providers_and_persists_the_default() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let (alpha_url, alpha_server) =
        start_provider_batch("200 OK", completion_response("alpha answer"), 2).await;
    let (beta_url, beta_server) =
        start_provider_batch("200 OK", completion_response("beta answer"), 2).await;
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    let config = format!(
        "# preserve this user comment\n[providers.alpha]\nbase_url = \"{alpha_url}\"\napi_key_env = \"D31_ALPHA_KEY\"\nmodels = [\"alpha-small\", \"alpha-large\"]\n\n[providers.beta]\nbase_url = \"{beta_url}\"\napi_key_env = \"D31_BETA_KEY\"\nmodels = [\"beta-chat\"]\n\n[selection]\ndefault_model = \"alpha/alpha-small\"\n"
    );
    fs::write(&config_path, config).expect("provider catalog is written");

    let output = run_repl(
        fixture.path(),
        &config_path,
        None,
        &format!(
            "/models\n/help {STRUCTURE_QUESTION}\n/model alpha/alpha-large\n/help {STRUCTURE_QUESTION}\n/model beta/beta-chat\n/help {STRUCTURE_QUESTION}\n/model alpha/alpha-large\n/models\n/model default beta/beta-chat\n/models\n/exit\n"
        ),
        &[
            ("D31_ALPHA_KEY", "alpha-secret"),
            ("D31_BETA_KEY", "beta-secret"),
        ],
    )
    .await;
    let alpha_requests = timeout(Duration::from_secs(5), alpha_server)
        .await
        .expect("REPL did not call alpha provider")
        .expect("alpha fake provider task completed");
    let saved_config = fs::read_to_string(&config_path).expect("updated config is readable");
    assert!(saved_config.contains("# preserve this user comment"));
    assert!(saved_config.contains("default_model = \"beta/beta-chat\""));
    let restart_output = run_cli_with_config(
        fixture.path(),
        &config_path,
        None,
        STRUCTURE_QUESTION,
        &[
            ("D31_ALPHA_KEY", "alpha-secret"),
            ("D31_BETA_KEY", "beta-secret"),
        ],
    )
    .await;
    let beta_requests = timeout(Duration::from_secs(5), beta_server)
        .await
        .expect("restarted CLI did not use persisted default")
        .expect("beta fake provider task completed");
    assert!(restart_output.status.success());

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("alpha/alpha-small [default, сессия]"));
    assert!(stdout.contains("alpha/alpha-large [сессия]"));
    assert!(stdout.contains("beta/beta-chat [default, сессия]"));
    assert!(stderr.is_empty());
    assert!(!stdout.contains("alpha-secret"));
    assert!(!stdout.contains("beta-secret"));
    assert_eq!(
        alpha_requests[0].request_line,
        "POST /v1/chat/completions HTTP/1.1"
    );

    let alpha_models = alpha_requests
        .iter()
        .map(|request| {
            serde_json::from_slice::<Value>(&request.body)
                .expect("alpha request is valid JSON")["model"]
                .as_str()
                .expect("alpha model is present")
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(alpha_models, ["alpha-small", "alpha-large"]);
    assert_eq!(
        beta_requests
            .iter()
            .map(|request| {
                serde_json::from_slice::<Value>(&request.body).expect("beta request is valid JSON")
                    ["model"]
                    .as_str()
                    .expect("beta model is present")
                    .to_owned()
            })
            .collect::<Vec<_>>(),
        ["beta-chat", "beta-chat"]
    );
    assert_eq!(
        alpha_requests[0].authorization.as_deref(),
        Some("Bearer alpha-secret")
    );
    assert_eq!(
        beta_requests[0].authorization.as_deref(),
        Some("Bearer beta-secret")
    );
    assert_eq!(
        beta_requests[1].authorization.as_deref(),
        Some("Bearer beta-secret")
    );
}

#[tokio::test]
async fn unknown_session_model_keeps_current_selection_and_saved_default() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    let config = "[providers.test]\nbase_url = \"http://127.0.0.1:1/v1\"\napi_key_env = \"D31_ALPHA_KEY\"\nmodels = [\"first-model\", \"second-model\"]\n\n[selection]\ndefault_model = \"test/first-model\"\n";
    fs::write(&config_path, config).expect("provider catalog is written");

    let output = run_repl(
        fixture.path(),
        &config_path,
        None,
        "/model test/second-model\n/model test/unlisted\n/models\n/exit\n",
        &[],
    )
    .await;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success());
    assert!(stdout.contains("Модель текущей сессии: test/second-model"));
    assert!(stdout.contains("test/first-model [default]"));
    assert!(stdout.contains("test/second-model [сессия]"));
    assert!(stderr.contains("Неизвестная пара provider/model"));
    assert!(!stderr.contains("unlisted"));
    assert_eq!(
        fs::read_to_string(&config_path).expect("config remains readable"),
        config
    );
}

#[tokio::test]
async fn startup_model_override_selects_without_changing_saved_default() {
    let fixture = git_project_with_docs(
        "# Modules\nThe project has core, web, persistence and transport modules.",
        "# Entry point\nThe primary web route is /agent.",
    );
    let (base_url, fake_provider) =
        start_provider("200 OK", completion_response("selected override")).await;
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    let config = format!(
        "[providers.test]\nbase_url = \"{base_url}\"\napi_key_env = \"D31_ALPHA_KEY\"\nmodels = [\"first-model\", \"second-model\"]\n\n[selection]\ndefault_model = \"test/first-model\"\n"
    );
    fs::write(&config_path, config).expect("provider catalog is written");

    let output = run_cli_with_config(
        fixture.path(),
        &config_path,
        Some("test/second-model"),
        STRUCTURE_QUESTION,
        &[("D31_ALPHA_KEY", "override-secret")],
    )
    .await;
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("one-shot CLI did not call fake provider")
        .expect("fake provider task completed");
    let request: Value = serde_json::from_slice(&request).expect("fake provider got valid JSON");
    let saved_config = fs::read_to_string(&config_path).expect("config remains readable");

    assert!(output.status.success());
    assert_eq!(request["model"], "second-model");
    assert!(saved_config.contains("default_model = \"test/first-model\""));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("override-secret"));
}

fn selected_project_or_fixture() -> (Option<TempDir>, PathBuf) {
    if let Some(root) = env::var_os("D31_TRANSACTION_IMPORT_ROOT") {
        return (None, PathBuf::from(root));
    }
    let fixture = git_project_with_docs(
        "# Transaction Import\n\n## Возможности\n\nThe project has core, transport, persistence and web modules. The read-only trace projects the next request without a new LLM call.",
        "# Entry point\nThe primary web route is /agent.",
    );
    fs::write(
        fixture.path().join("docs/data-api.md"),
        "# Schema\nThe transaction-import.v3 schema describes structured import data.",
    )
    .unwrap();
    git_output(fixture.path(), &["add", "docs/data-api.md"]);
    let root = fixture.path().to_owned();
    (Some(fixture), root)
}

fn empty_git_project() -> TempDir {
    let fixture = tempfile::tempdir().unwrap();
    initialize_git(fixture.path());
    fixture
}

fn git_project_with_docs(readme: &str, architecture: &str) -> TempDir {
    let fixture = tempfile::tempdir().unwrap();
    fs::create_dir_all(fixture.path().join("docs")).unwrap();
    fs::write(fixture.path().join("README.md"), readme).unwrap();
    fs::write(fixture.path().join("docs/architecture.md"), architecture).unwrap();
    initialize_git(fixture.path());
    git_output(fixture.path(), &["add", "README.md", "docs"]);
    fixture
}

fn initialize_git(root: &Path) {
    let output = StdCommand::new("git")
        .arg("init")
        .arg("--initial-branch=main")
        .current_dir(root)
        .output()
        .expect("git is installed");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn git_output(root: &Path, args: &[&str]) -> String {
    let output = StdCommand::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .expect("git is installed");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

async fn run_cli(
    project_root: &Path,
    question: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Output {
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    let base_url = base_url.unwrap_or("http://127.0.0.1:1/v1");
    let config = format!(
        "[providers.test]\nbase_url = \"{base_url}\"\napi_key_env = \"DEVELOPER_ASSISTANT_API_KEY\"\nmodels = [\"gpt-4o-mini\"]\n\n[selection]\ndefault_model = \"test/gpt-4o-mini\"\n"
    );
    fs::write(&config_path, config).expect("test config is written");

    let mut command = Command::new(env!("CARGO_BIN_EXE_developer-assistant"));
    command
        .current_dir(project_root)
        .arg("--config")
        .arg(&config_path)
        .arg("/help")
        .arg(question)
        .env_remove("DEVELOPER_ASSISTANT_API_KEY")
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL");
    if let Some(api_key) = api_key {
        command.env("DEVELOPER_ASSISTANT_API_KEY", api_key);
    }
    command
        .output()
        .await
        .expect("developer-assistant binary starts")
}

async fn run_cli_with_git_index(
    project_root: &Path,
    question: &str,
    base_url: &str,
    api_key: &str,
    alternate_index: &Path,
) -> Output {
    let config_directory = TempDir::new().expect("temporary config directory");
    let config_path = config_directory.path().join("config.toml");
    fs::write(
        &config_path,
        format!(
            "[providers.test]\nbase_url = \"{base_url}\"\napi_key_env = \"DEVELOPER_ASSISTANT_API_KEY\"\nmodels = [\"gpt-4o-mini\"]\n\n[selection]\ndefault_model = \"test/gpt-4o-mini\"\n"
        ),
    )
    .expect("test config is written");

    Command::new(env!("CARGO_BIN_EXE_developer-assistant"))
        .current_dir(project_root)
        .arg("--config")
        .arg(config_path)
        .arg("/help")
        .arg(question)
        .env("DEVELOPER_ASSISTANT_API_KEY", api_key)
        .env("GIT_INDEX_FILE", alternate_index)
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL")
        .output()
        .await
        .expect("developer-assistant binary starts")
}

async fn run_cli_with_config(
    project_root: &Path,
    config_path: &Path,
    model_override: Option<&str>,
    question: &str,
    environment: &[(&str, &str)],
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_developer-assistant"));
    command
        .current_dir(project_root)
        .arg("--config")
        .arg(config_path)
        .env_remove("DEVELOPER_ASSISTANT_API_KEY")
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL")
        .env_remove("D31_ALPHA_KEY")
        .env_remove("D31_BETA_KEY");
    if let Some(model) = model_override {
        command.arg("--model").arg(model);
    }
    command.arg("/help").arg(question);
    for (name, value) in environment {
        command.env(name, value);
    }
    command
        .output()
        .await
        .expect("developer-assistant binary starts")
}

async fn run_repl(
    project_root: &Path,
    config_path: &Path,
    model_override: Option<&str>,
    input: &str,
    environment: &[(&str, &str)],
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_developer-assistant"));
    command
        .current_dir(project_root)
        .arg("--config")
        .arg(config_path)
        .env_remove("DEVELOPER_ASSISTANT_API_KEY")
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL")
        .env_remove("D31_ALPHA_KEY")
        .env_remove("D31_BETA_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(model) = model_override {
        command.arg("--model").arg(model);
    }
    for (name, value) in environment {
        command.env(name, value);
    }
    let mut child = command.spawn().expect("developer-assistant binary starts");
    let mut stdin = child.stdin.take().expect("REPL stdin is piped");
    stdin
        .write_all(input.as_bytes())
        .await
        .expect("REPL input is written");
    drop(stdin);
    child
        .wait_with_output()
        .await
        .expect("REPL process completes")
}

fn user_prompt(request: &Value) -> &str {
    request["messages"]
        .as_array()
        .expect("chat messages array")
        .iter()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .expect("user prompt exists")
}

async fn start_provider(
    status: &'static str,
    response_body: String,
) -> (String, JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request_body = read_request_body(&mut stream).await;
        let response = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response_body.len(),
            response_body,
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        request_body
    });
    (format!("http://{address}/v1"), server)
}

async fn start_provider_batch(
    status: &'static str,
    response_body: String,
    request_count: usize,
) -> (String, JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut requests = Vec::with_capacity(request_count);
        for _ in 0..request_count {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_complete_request(&mut stream).await;
            let captured = capture_request(request);
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
            requests.push(captured);
        }
        requests
    });
    (format!("http://{address}/v1"), server)
}

async fn start_provider_script(
    response_bodies: Vec<String>,
) -> (String, JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut requests = Vec::with_capacity(response_bodies.len());
        for response_body in response_bodies {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = capture_request(read_complete_request(&mut stream).await);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
            requests.push(request);
        }
        requests
    });
    (format!("http://{address}/v1"), server)
}

async fn start_provider_and_update_readme(
    project_root: PathBuf,
    response_bodies: Vec<String>,
) -> (String, JoinHandle<Vec<CapturedRequest>>) {
    assert_eq!(response_bodies.len(), 2);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut requests = Vec::with_capacity(response_bodies.len());
        for (index, response_body) in response_bodies.into_iter().enumerate() {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = capture_request(read_complete_request(&mut stream).await);
            if index == 0 {
                fs::write(
                    project_root.join("README.md"),
                    "# Configuration\nThe configured storage engine uses NEW_ENGINE_MARKER.\n",
                )
                .unwrap();
                git_output(&project_root, &["add", "README.md"]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
            requests.push(request);
        }
        requests
    });
    (format!("http://{address}/v1"), server)
}

async fn start_redirect_provider() -> (String, JoinHandle<Vec<u8>>, JoinHandle<()>) {
    let target_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target_address = target_listener.local_addr().unwrap();
    let target_server = tokio::spawn(async move {
        let (mut stream, _) = target_listener.accept().await.unwrap();
        let mut request = [0_u8; 1_024];
        let _ = stream.read(&mut request).await.unwrap();
    });

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request_body = read_request_body(&mut stream).await;
        let response = format!(
            "HTTP/1.1 302 Found\r\nlocation: http://{target_address}/capture\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        request_body
    });
    (format!("http://{address}/v1"), server, target_server)
}

async fn read_request_body(stream: &mut TcpStream) -> Vec<u8> {
    capture_request(read_complete_request(stream).await).body
}

async fn read_complete_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0_u8; 1_024];
        let bytes_read = stream.read(&mut chunk).await.unwrap();
        assert!(bytes_read > 0, "provider request ended before its body");
        request.extend_from_slice(&chunk[..bytes_read]);

        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .expect("reqwest sends Content-Length for the JSON request");
        if request.len() >= header_end + 4 + content_length {
            return request;
        }
    }
}

fn capture_request(request: Vec<u8>) -> CapturedRequest {
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP request has headers");
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let authorization = headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("authorization")
            .then(|| value.trim().to_owned())
    });
    let body_start = header_end + 4;
    CapturedRequest {
        request_line: headers.lines().next().expect("request line").to_owned(),
        body: request[body_start..].to_vec(),
        authorization,
    }
}

fn tool_call_response(id: &str, name: &str, arguments: Value, uncached_tokens: u64) -> String {
    json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments.to_string()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": uncached_tokens,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": uncached_tokens
            }
        }
    })
    .to_string()
}

fn completion_response(content: &str) -> String {
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10
            }
        }
    })
    .to_string()
}
