use serde_json::{Value, json};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output},
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

#[tokio::test]
async fn help_uses_docs_and_live_mcp_context() {
    let (_fixture, root) = selected_project_or_fixture();
    let expected_branch = git_output(&root, &["branch", "--show-current"]);
    let before = git_output(&root, &["status", "--porcelain", "--untracked-files=all"]);
    let response = completion_response(STRUCTURE_ANSWER);
    let (base_url, fake_provider) = start_provider("200 OK", response).await;

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
    let request = timeout(Duration::from_secs(5), fake_provider)
        .await
        .expect("CLI did not call the local fake provider")
        .expect("fake provider task failed");
    let after = git_output(&root, &["status", "--porcelain", "--untracked-files=all"]);

    assert_eq!(
        before, after,
        "CLI changed the selected project working tree"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("Ветка Git: `{expected_branch}`")));
    assert!(stdout.contains(STRUCTURE_ANSWER));
    assert!(stdout.contains("docs/architecture.md"));
    assert!(stdout.contains("Источники:"));

    let request: Value = serde_json::from_slice(&request).expect("fake provider got valid JSON");
    assert_eq!(request["model"], "gpt-4o-mini");
    assert_eq!(request["temperature"], 0.0);
    assert_eq!(request["max_completion_tokens"], 800);
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
    assert!(prompt.contains("[docs/data-api.md#"));
    assert!(prompt.contains("core"));
    assert!(!prompt.contains(TEST_API_KEY));
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

    assert!(prompt.contains("[README.md#Возможности]"));
    assert!(prompt.contains("read-only trace"));
    assert!(String::from_utf8_lossy(&output.stdout).contains("README.md"));
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
    assert!(stdout.contains("В документации нет подходящего контекста"));
    assert!(stdout.contains("provider не вызывался"));
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
        fs::write(fixture.path().join(format!("a-{index:02}.txt")), "baseline").unwrap();
    }
    fs::write(fixture.path().join("conflict.txt"), "baseline").unwrap();
    fs::write(fixture.path().join("z-last.txt"), "baseline").unwrap();
    git_output(fixture.path(), &["add", "."]);
    git_output(fixture.path(), &["commit", "-m", "baseline"]);

    git_output(fixture.path(), &["switch", "-c", "other"]);
    fs::write(fixture.path().join("conflict.txt"), "other branch").unwrap();
    git_output(fixture.path(), &["add", "conflict.txt"]);
    git_output(fixture.path(), &["commit", "-m", "other conflict change"]);

    git_output(fixture.path(), &["switch", "main"]);
    fs::write(fixture.path().join("conflict.txt"), "main branch").unwrap();
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

    assert!(stdout.contains("Первая строка.\n\\u{1b}[2J\\u{7}\\u{202e}готово"));
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
            json!({"choices": [{"message": {"role": "assistant", "content": "  "}}]}).to_string(),
            "пустой ответ",
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_developer-assistant"));
    command
        .arg("--project-root")
        .arg(project_root)
        .arg("/help")
        .arg(question)
        .env_remove("DEVELOPER_ASSISTANT_API_KEY")
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL");
    if let Some(base_url) = base_url {
        command.env("DEVELOPER_ASSISTANT_BASE_URL", base_url);
    }
    if let Some(api_key) = api_key {
        command.env("DEVELOPER_ASSISTANT_API_KEY", api_key);
    }
    command
        .env("DEVELOPER_ASSISTANT_MODEL", "gpt-4o-mini")
        .output()
        .await
        .expect("developer-assistant binary starts")
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
        let body_start = header_end + 4;
        if request.len() >= body_start + content_length {
            return request[body_start..body_start + content_length].to_vec();
        }
    }
}

fn completion_response(content: &str) -> String {
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
    .to_string()
}
