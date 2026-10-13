use anyhow::Result;
use developer_assistant::review::deepseek::{DeepSeekClient, InputTokenUsage, ReviewToolExecutor};
use developer_assistant::review::{
    corpus::load_project_corpus,
    embedding::{EMBEDDING_DIMENSIONS, Embedder, FastEmbedder},
    index::ReviewIndex,
};
use developer_assistant::review::{
    github::{GitHubClient, PullRequestId},
    report::{FindingCategory, ReviewFinding, finding_fingerprint},
    runner::ReviewRunner,
};
use reqwest::Url;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    path::Path,
    process::{Command, Output},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

#[derive(Clone, Default)]
struct FixtureEmbedder {
    embedded_inputs: Arc<AtomicUsize>,
}

impl Embedder for FixtureEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedded_inputs
            .fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts
            .iter()
            .map(|text| {
                let lower = text.to_lowercase();
                let dimension = if lower.contains("transaction")
                    || lower.contains("balance")
                    || lower.contains("settlement")
                {
                    0
                } else {
                    1
                };
                let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
                embedding[dimension] = 1.0;
                embedding
            })
            .collect())
    }
}
#[derive(Default)]
struct OrthogonalEmbedder;

impl Embedder for OrthogonalEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let dimension = usize::from(text.to_ascii_lowercase().contains("orthogonal query"));
                let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
                embedding[dimension] = 1.0;
                embedding
            })
            .collect())
    }
}

#[test]
fn d32_review_filtered_rag() {
    let fixture = project_fixture();
    let root = fixture.path();
    let corpus = load_project_corpus(root).expect("filtered corpus loads");
    let paths = corpus
        .iter()
        .map(|document| document.path.as_str())
        .collect::<HashSet<_>>();

    assert!(paths.contains("README.md"));
    assert!(paths.contains("docs/architecture.md"));
    assert!(paths.contains("src/transaction.rs"));
    for excluded in [
        "ignored.txt",
        "src/config.rs",
        "credentials.txt",
        "tokens.json",
        "settings.json",
        "src/auth.rs",
        "src/google_config.rs",
        "build/generated.rs",
        "vendor/library.rs",
        "oversized.txt",
        "src/binary.dat",
        "src/non-utf8.txt",
        "src/alias.rs",
    ] {
        assert!(
            !paths.contains(excluded),
            "filtered corpus included {excluded}"
        );
    }

    let cache = root.join(".data/developer-assistant");
    let mut index = ReviewIndex::open(root, &cache, FixtureEmbedder::default())
        .expect("review index builds with a deterministic test embedder");
    let results = index
        .search("How does the importer validate transaction balances?", 8)
        .expect("hybrid retrieval succeeds");
    let result_paths = results
        .iter()
        .map(|result| result.path.as_str())
        .collect::<HashSet<_>>();
    assert!(result_paths.contains("docs/architecture.md"));
    assert!(result_paths.contains("src/transaction.rs"));
    assert!(!result_paths.contains("ignored.txt"));

    let excerpt = index
        .read_file("src/transaction.rs", 1, 2)
        .expect("allowlisted project file can be read");
    assert!(excerpt.contains("validate_transaction_balance"));
    assert!(excerpt.starts_with("    1 | "));
    assert!(index.read_file("../README.md", 1, 2).is_err());
    assert!(index.read_file("ignored.txt", 1, 2).is_err());
    assert!(index.read_file("src/transaction.rs", 1, 81).is_err());
}

#[test]
fn d32_review_lexical_identifier_retrieval() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".data/lexical-review");
    let mut index = ReviewIndex::open(root, &cache, OrthogonalEmbedder)
        .expect("review index builds with orthogonal test vectors");
    let results = index
        .search("orthogonal query validate_transaction_balance", 8)
        .expect("lexical retrieval handles an identifier absent from semantic vectors");

    assert_eq!(
        results.first().map(|result| result.path.as_str()),
        Some("src/transaction.rs")
    );
}

#[cfg(unix)]
#[test]
fn d32_review_cache_rejects_symlink_escape() {
    let fixture = project_fixture();
    let root = fixture.path();
    let outside = tempfile::tempdir().expect("outside directory is created");
    fs::create_dir(root.join(".data")).expect("local cache parent is created");
    std::os::unix::fs::symlink(outside.path(), root.join(".data/external"))
        .expect("cache escape symlink is created");

    let cache_dir = root.join(".data/external");
    let index_result = ReviewIndex::open(root, &cache_dir, OrthogonalEmbedder);
    let model_result = FastEmbedder::load(root, &cache_dir);
    assert!(index_result.is_err());
    assert!(model_result.is_err());
    assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
}

#[test]
fn d32_review_index_cache_matches_current_corpus() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".data/developer-assistant");
    let embedder = FixtureEmbedder::default();
    let count = Arc::clone(&embedder.embedded_inputs);

    let first = ReviewIndex::open(root, &cache, embedder.clone()).expect("initial index builds");
    let initial_embeddings = count.load(Ordering::Relaxed);
    assert!(initial_embeddings > 0);
    drop(first);

    let second = ReviewIndex::open(root, &cache, embedder.clone()).expect("matching cache loads");
    assert_eq!(count.load(Ordering::Relaxed), initial_embeddings);
    drop(second);

    write_text(
        root,
        "docs/architecture.md",
        "The transaction importer validates each account balance before producing its daily batch.\nThe revised pipeline checks duplicate transaction identifiers before persistence.\n",
    );
    let _updated =
        ReviewIndex::open(root, &cache, embedder).expect("changed corpus rebuilds index");
    assert!(count.load(Ordering::Relaxed) > initial_embeddings);
}

#[test]
fn d32_review_ignored_paths_do_not_deadlock() {
    let fixture = project_fixture();
    let root = fixture.path();
    write_text(root, ".gitignore", "ignored.txt\n.data/\nignored-many/\n");
    for index in 0..512 {
        let path = format!("ignored-many/{index:04}-{}.txt", "x".repeat(180));
        write_text(root, &path, "synthetic ignored content\n");
    }

    let add = git(root, &["add", "-f", "-A"]);
    assert!(
        add.status.success(),
        "git add failed: {}",
        String::from_utf8_lossy(&add.stderr)
    );
    let corpus = load_project_corpus(root).expect("many ignored paths are processed concurrently");
    assert!(corpus.iter().any(|document| document.path == "README.md"));
    assert!(
        corpus
            .iter()
            .all(|document| !document.path.starts_with("ignored-many/"))
    );
}

#[test]
fn d32_review_cache_requires_git_exclusion() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".review-cache");

    assert!(ReviewIndex::open(root, &cache, OrthogonalEmbedder).is_err());
    assert!(FastEmbedder::load(root, &cache).is_err());
    assert!(!cache.exists(), "unignored cache directory is not created");
}
fn project_fixture() -> TempDir {
    let fixture = tempfile::tempdir().expect("temporary project directory is created");
    let root = fixture.path();
    let init = git(root, &["init", "-q"]);
    assert!(
        init.status.success(),
        "git init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    write_text(root, ".gitignore", "ignored.txt\n.data/\n");
    write_text(
        root,
        "README.md",
        "Project overview and module boundaries.\n",
    );
    write_text(
        root,
        "docs/architecture.md",
        "The transaction importer validates each account balance before producing its daily batch.\n",
    );
    write_text(
        root,
        "src/transaction.rs",
        "pub fn validate_transaction_balance() {\n    validate_account_balance();\n}\n",
    );
    write_text(
        root,
        "docs/login.md",
        "A session cookie is created after a successful login.\n",
    );
    write_text(
        root,
        "ignored.txt",
        "transaction balance validation is documented here.\n",
    );
    write_text(
        root,
        "src/config.rs",
        "const API_KEY: &str = \"ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";\n",
    );
    write_text(
        root,
        "credentials.txt",
        "A credential file must never enter the project index.\n",
    );
    write_text(
        root,
        "tokens.json",
        "{\"token\":\"synthetic fixture value\"}\n",
    );
    write_text(
        root,
        "settings.json",
        r#"{"api_key":"0123456789abcdef0123456789abcdef"}"#,
    );
    write_text(
        root,
        "src/auth.rs",
        "Authorization: Bearer synthetic_token_value_0123456789abcdef\n",
    );
    write_text(
        root,
        "src/google_config.rs",
        "const GOOGLE_MAPS_API_KEY: &str = \"0123456789abcdef0123456789abcdef\";\nconst RAW: &str = \"AIza0123456789abcdefghijklmnopqrstuvwxyzAB\";\n",
    );
    write_text(root, "build/generated.rs", "generated transaction helper\n");
    write_text(root, "vendor/library.rs", "vendor transaction helper\n");
    write_text(root, "oversized.txt", &"x".repeat(256 * 1024 + 1));
    fs::write(root.join("src/binary.dat"), [0_u8, 1, 2, 3, 4]).expect("binary fixture is written");
    fs::write(root.join("src/non-utf8.txt"), [0xff, 0xfe, 0x80])
        .expect("non-UTF8 fixture is written");
    #[cfg(unix)]
    std::os::unix::fs::symlink("transaction.rs", root.join("src/alias.rs"))
        .expect("symlink fixture is created");

    let output = git(root, &["add", "-f", "-A"]);
    assert!(
        output.status.success(),
        "git add failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fixture
}

fn write_text(root: &Path, relative: &str, text: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("fixture path has a parent"))
        .expect("fixture parent is created");
    fs::write(path, text).expect("fixture file is written");
}

fn git(root: &Path, arguments: &[&str]) -> Output {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .expect("git process starts")
}
#[derive(Default)]
struct CountingTools {
    calls: usize,
}

impl ReviewToolExecutor for CountingTools {
    fn execute(&mut self, _name: &str, _arguments: Value) -> Result<String> {
        self.calls += 1;
        Ok("{\"ok\":true}".to_owned())
    }
}

#[derive(Clone, Debug)]
struct CapturedRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Value,
}

#[tokio::test]
async fn d32_review_token_budget_denies_pending_tool_and_logs_final_usage_separately() {
    let tool_call = json!({
        "id": "call_budget_1",
        "type": "function",
        "function": {
            "name": "search_code",
            "arguments": "{\"query\":\"transaction totals\",\"limit\":2}"
        }
    });
    let responses = vec![
        deepseek_response(
            "tool_calls",
            json!({"role":"assistant","content":null,"tool_calls":[tool_call]}),
            100,
            60,
            40,
        ),
        deepseek_response(
            "stop",
            json!({
                "role": "assistant",
                "content": "{\"summary\":\"Review is bounded.\",\"findings\":[],\"resolved_findings\":[]}"
            }),
            100,
            80,
            20,
        ),
    ];
    let (endpoint, server) = scripted_http(responses).await;
    let client = DeepSeekClient::new(
        endpoint,
        "synthetic-review-key".to_owned(),
        "deepseek-flash".to_owned(),
    )
    .expect("loopback fake endpoint is accepted");
    let mut tools = CountingTools::default();

    let result = client
        .review("Only review.", "Synthetic PR diff.", &mut tools, 40)
        .await
        .expect("token-budget fallback returns final response");
    let requests = server.await.expect("fake DeepSeek server completed");

    assert_eq!(tools.calls, 0, "pending tool is not run at the budget");
    assert_eq!(result.retrieval_uncached_input_tokens, 40);
    assert_eq!(
        result.final_usage,
        Some(InputTokenUsage {
            prompt_tokens: 100,
            cached_input_tokens: 80,
            uncached_input_tokens: 20,
        })
    );
    assert_eq!(result.requests, 2);
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer synthetic-review-key")
    );
    assert_eq!(requests[0].body["thinking"]["type"], "disabled");
    assert_eq!(requests[0].body["tool_choice"], "auto");
    assert_eq!(requests[0].body["tools"].as_array().unwrap().len(), 2);
    assert_eq!(requests[1].body["tool_choice"], "none");
    assert!(requests[1].body.get("tools").is_none());
    assert_eq!(requests[1].body["response_format"]["type"], "json_object");

    let tool_message = requests[1].body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("budget result is returned as a tool message");
    assert_eq!(tool_message["tool_call_id"], "call_budget_1");
    let tool_content: Value =
        serde_json::from_str(tool_message["content"].as_str().unwrap()).unwrap();
    assert_eq!(tool_content["error"], "input_budget_exhausted");
}

#[tokio::test]
async fn d32_review_zero_uncached_usage_stops_before_tool_execution() {
    let tool_call = json!({
        "id": "call_zero_usage",
        "type": "function",
        "function": {
            "name": "search_code",
            "arguments": "{\"query\":\"transaction totals\",\"limit\":2}"
        }
    });
    let (endpoint, server) = scripted_http(vec![deepseek_response(
        "tool_calls",
        json!({"role":"assistant","content":null,"tool_calls":[tool_call]}),
        0,
        0,
        0,
    )])
    .await;
    let client = DeepSeekClient::new(
        endpoint,
        "synthetic-review-key".to_owned(),
        "deepseek-flash".to_owned(),
    )
    .expect("loopback fake endpoint is accepted");
    let mut tools = CountingTools::default();

    let error = client
        .review("Only review.", "Synthetic PR diff.", &mut tools, 100)
        .await
        .expect_err("zero-token tool loop cannot advance its budget");
    let requests = server.await.expect("fake DeepSeek server completed");

    assert!(error.to_string().contains("uncached input token budget"));
    assert_eq!(tools.calls, 0);
    assert_eq!(requests.len(), 1);
}

fn deepseek_response(
    finish_reason: &str,
    message: Value,
    prompt_tokens: u64,
    cache_hit_tokens: u64,
    cache_miss_tokens: u64,
) -> Value {
    json!({
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": cache_hit_tokens,
                "prompt_cache_miss_tokens": cache_miss_tokens
            }
        }
    })
}

async fn scripted_http(responses: Vec<Value>) -> (Url, JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("fake endpoint binds");
    let address = listener.local_addr().expect("fake endpoint has address");
    let endpoint = Url::parse(&format!("http://{address}/v1/chat/completions"))
        .expect("fake endpoint URL is valid");
    let server = tokio::spawn(async move {
        let mut captured = Vec::with_capacity(responses.len());
        for response in responses {
            let (mut stream, _) = listener.accept().await.expect("client connects");
            captured.push(read_http_request(&mut stream).await);
            let body = serde_json::to_vec(&response).expect("response serializes");
            let mut header = Vec::new();
            header.extend_from_slice(b"HTTP/1.1 200 OK");
            header.extend_from_slice(&[13, 10]);
            header.extend_from_slice(b"Content-Type: application/json");
            header.extend_from_slice(&[13, 10]);
            header.extend_from_slice(format!("Content-Length: {}", body.len()).as_bytes());
            header.extend_from_slice(&[13, 10]);
            header.extend_from_slice(b"Connection: close");
            header.extend_from_slice(&[13, 10, 13, 10]);
            stream.write_all(&header).await.expect("header writes");
            stream.write_all(&body).await.expect("body writes");
        }
        captured
    });
    (endpoint, server)
}

async fn read_http_request(stream: &mut TcpStream) -> CapturedRequest {
    let mut request = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream.read(&mut chunk).await.expect("request reads");
        assert!(
            read > 0,
            "request closes before Content-Length is satisfied"
        );
        request.extend_from_slice(&chunk[..read]);
        let Some(header_end) = request
            .windows(4)
            .position(|window| window == [13_u8, 10, 13, 10])
        else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let mut request_line = headers.lines().next().unwrap().split_whitespace();
        let method = request_line.next().expect("request has method").to_owned();
        let path = request_line.next().expect("request has path").to_owned();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length").then(|| {
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("Content-Length parses")
                })
            })
            .unwrap_or(0);
        if request.len() < header_end + 4 + content_length {
            continue;
        }

        let authorization = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("authorization")
                .then(|| value.trim().to_owned())
        });
        let body_start = header_end + 4;
        let body = if content_length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&request[body_start..body_start + content_length])
                .expect("request body is JSON")
        };
        return CapturedRequest {
            method,
            path,
            authorization,
            body,
        };
    }
}

#[tokio::test]
async fn d32_review_pipeline_is_idempotent_and_resolves_only_with_changed_head_evidence() {
    let fixture = review_project_fixture();
    let root = fixture.path();
    let base_sha = "b".repeat(40);
    let head_one = git_head(root);
    let patch_one =
        "@@ -0,0 +1,3 @@\n+pub fn calculate() {\n+    calculate_transaction_total();\n+}\n";
    let (github_url, github_server, control) = mock_github_server(
        base_sha.clone(),
        head_one.clone(),
        patch_one.to_owned(),
        None,
    )
    .await;
    let findings = review_findings();
    let finding = &findings[0];
    let fingerprint = finding_fingerprint(finding);
    let tool_call = json!({
        "id": "call_initial_rag",
        "type": "function",
        "function": {
            "name": "search_code",
            "arguments": "{\"query\":\"transaction balance contract\",\"limit\":3}"
        }
    });
    let responses = vec![
        deepseek_response(
            "tool_calls",
            json!({"role":"assistant","content":null,"tool_calls":[tool_call]}),
            1_000,
            700,
            300,
        ),
        deepseek_review_response(review_report_many(
            "Проверка баланса отсутствует перед расчетом итоговой суммы; архитектуру и рекомендации тоже проверяем.",
            &findings,
            Vec::new(),
        )),
        deepseek_review_response(review_report_many(
            "Повторная проверка подтверждает те же категории и места.",
            &findings,
            Vec::new(),
        )),
        deepseek_review_response(review_report(
            "В новом head добавлена проверка баланса до расчета.",
            None,
            vec![json!({
                "fingerprint": fingerprint,
                "path": "src/transaction.rs",
                "evidence": "    if !validate_transaction_balance() {"
            })],
        )),
    ];
    let (deepseek_url, deepseek_server) = scripted_http(responses).await;
    let github = GitHubClient::new(github_url, "synthetic-github-token".to_owned())
        .expect("fake GitHub endpoint is accepted");
    let deepseek = DeepSeekClient::new(
        deepseek_url,
        "synthetic-deepseek-key".to_owned(),
        "deepseek-flash".to_owned(),
    )
    .expect("fake DeepSeek endpoint is accepted");
    let runner = ReviewRunner::new(github, deepseek);
    let identity_one = PullRequestId {
        owner: "acme".to_owned(),
        repository: "transaction-import".to_owned(),
        number: 7,
        expected_base_sha: base_sha.clone(),
        expected_head_sha: head_one,
    };
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-pipeline/index"),
        FixtureEmbedder::default(),
    )
    .expect("synthetic corpus indexes without a network model");

    let first = runner
        .run_with_index(root, &identity_one, 10_000, &mut index)
        .await
        .expect("first review publishes all three required finding categories");
    assert_eq!(first.findings_published, 3);
    assert_eq!(first.threads_created_or_updated, 3);
    assert_eq!(first.threads_resolved, 0);
    assert_eq!(first.retrieval_uncached_input_tokens, 600);
    assert_eq!(first.tool_calls_executed, 1);
    assert_eq!(first.requests, 2);

    let secret = "AIza0123456789abcdefghijklmnopqrstuvwxyzAB";
    let edited_prior_body = format!(
        "<!-- developer-assistant:finding:v1:{fingerprint} -->\nPrior body edited with GOOGLE_MAPS_API_KEY={secret}\nEvidence:\nold evidence"
    );
    update_mock_thread(&control, 500, edited_prior_body, true).await;

    let second = runner
        .run_with_index(root, &identity_one, 10_000, &mut index)
        .await
        .expect("repeated review refreshes current anchors and findings");
    assert_eq!(second.findings_published, 3);
    assert_eq!(second.threads_created_or_updated, 3);

    write_text(
        root,
        "src/transaction.rs",
        "pub fn calculate() {\n    if !validate_transaction_balance() {\n        return;\n    }\n    calculate_transaction_total();\n}\n",
    );
    let head_two = commit_fixture(root, "add balance guard");
    let patch_two = "@@ -0,0 +1,6 @@\n+pub fn calculate() {\n+    if !validate_transaction_balance() {\n+        return;\n+    }\n+    calculate_transaction_total();\n+}\n";
    set_mock_snapshot(&control, head_two.clone(), patch_two.to_owned()).await;
    drop(index);
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-pipeline/index"),
        FixtureEmbedder::default(),
    )
    .expect("updated head rebuilds its local index");
    let identity_two = PullRequestId {
        expected_head_sha: head_two,
        ..identity_one
    };
    let third = runner
        .run_with_index(root, &identity_two, 10_000, &mut index)
        .await
        .expect("resolution with current changed-code evidence succeeds");
    assert_eq!(third.findings_published, 0);
    assert_eq!(third.threads_resolved, 2);

    let deepseek_requests = deepseek_server
        .await
        .expect("scripted DeepSeek server handled all requests");
    assert_eq!(deepseek_requests.len(), 4);
    let first_user_prompt = request_user_json(&deepseek_requests[0]);
    assert_eq!(
        first_user_prompt["changed_files"],
        json!(["src/transaction.rs", "src/fees.rs"])
    );
    let diff = first_user_prompt["diff"].as_str().unwrap();
    let expected_fee_patch = "@@ -0,0 +1,4 @@\n+pub fn calculate_fee() {\n+    calculate_transaction_total();\n+    write_fee_audit_log();\n+}\n";
    let expected_diff = format!(
        "--- changed file: src/transaction.rs ---\n{patch_one}\n--- changed file: src/fees.rs ---\n{expected_fee_patch}\n"
    );
    assert_eq!(diff, expected_diff);
    assert!(!diff.contains("assets/logo.png"));
    assert!(
        !first_user_prompt["initial_retrieval"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        first_user_prompt["initial_retrieval"]
            .to_string()
            .contains("docs/architecture.md")
    );
    let second_run_prompt = request_user_json(&deepseek_requests[2]);
    assert_eq!(
        second_run_prompt["active_prior_findings"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(
        !second_run_prompt
            .to_string()
            .contains("AIza0123456789abcdefghijklmnopqrstuvwxyzAB")
    );
    let first_tool_result = deepseek_requests[1].body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool retrieval result is part of the next model request");
    assert!(
        first_tool_result["content"]
            .as_str()
            .unwrap()
            .contains("docs/architecture.md")
    );
    assert!(
        !first_tool_result["content"]
            .as_str()
            .unwrap()
            .contains("credentials.txt")
    );
    assert!(
        deepseek_requests.iter().all(
            |request| request.authorization.as_deref() == Some("Bearer synthetic-deepseek-key")
        )
    );

    control
        .send(MockGithubControl::Shutdown)
        .expect("fake GitHub server accepts shutdown");
    let github_state = github_server
        .await
        .expect("fake GitHub server stopped cleanly");
    assert_eq!(
        github_state
            .threads
            .iter()
            .filter(|thread| !thread.is_resolved)
            .count(),
        2,
        "unchanged architecture and recommendation findings stay open"
    );
    assert_eq!(github_state.threads.len(), 4);
    assert_eq!(github_state.summaries.len(), 1);
    assert!(
        github_state
            .threads
            .iter()
            .any(|thread| thread.body.contains("[bug]"))
    );
    for category in ["[architecture]", "[recommendation]"] {
        assert!(
            github_state
                .threads
                .iter()
                .any(|thread| !thread.is_resolved && thread.body.contains(category))
        );
    }
    let first_summary = github_state
        .requests
        .iter()
        .find(|request| {
            request.method == "POST"
                && request.path == "/api/repos/acme/transaction-import/issues/7/comments"
        })
        .expect("initial summary is published")
        .body["body"]
        .as_str()
        .unwrap();
    for heading in [
        "## Потенциальные баги",
        "## Архитектурные проблемы",
        "## Рекомендации",
    ] {
        assert!(first_summary.contains(heading));
    }
    assert_eq!(
        github_state
            .requests
            .iter()
            .filter(|request| request.method == "POST"
                && request.path == "/api/repos/acme/transaction-import/pulls/7/comments")
            .count(),
        4,
        "a new current anchor replaces only the outdated inline thread"
    );
    assert_eq!(
        github_state
            .requests
            .iter()
            .filter(|request| request.method == "PATCH"
                && request
                    .path
                    .starts_with("/api/repos/acme/transaction-import/pulls/comments/"))
            .count(),
        2
    );
    assert_eq!(
        github_state
            .requests
            .iter()
            .filter(|request| request.method == "POST"
                && request.body["query"]
                    .as_str()
                    .is_some_and(|query| query.contains("resolveReviewThread")))
            .count(),
        2
    );
    assert!(
        github_state
            .threads
            .iter()
            .filter(|thread| thread.is_resolved)
            .all(|thread| thread.body.contains(&fingerprint))
    );
    assert!(
        github_state.requests.iter().all(
            |request| request.authorization.as_deref() == Some("Bearer synthetic-github-token")
        )
    );
}

#[tokio::test]
async fn d32_review_stale_head_performs_no_github_writes() {
    let fixture = review_project_fixture();
    let root = fixture.path();
    let base_sha = "b".repeat(40);
    let head_sha = git_head(root);
    let patch = "@@ -0,0 +1,3 @@\n+pub fn calculate() {\n+    calculate_transaction_total();\n+}\n";
    let (github_url, github_server, control) = mock_github_server(
        base_sha.clone(),
        head_sha.clone(),
        patch.to_owned(),
        Some(2),
    )
    .await;
    let (deepseek_url, deepseek_server) =
        scripted_http(vec![deepseek_review_response(review_report(
            "Potential issue in changed total calculation.",
            Some(&review_finding()),
            Vec::new(),
        ))])
        .await;
    let github = GitHubClient::new(github_url, "synthetic-github-token".to_owned()).unwrap();
    let deepseek = DeepSeekClient::new(
        deepseek_url,
        "synthetic-deepseek-key".to_owned(),
        "deepseek-flash".to_owned(),
    )
    .unwrap();
    let runner = ReviewRunner::new(github, deepseek);
    let identity = PullRequestId {
        owner: "acme".to_owned(),
        repository: "transaction-import".to_owned(),
        number: 7,
        expected_base_sha: base_sha,
        expected_head_sha: head_sha,
    };
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-stale/index"),
        FixtureEmbedder::default(),
    )
    .expect("synthetic corpus indexes");

    let result = runner
        .run_with_index(root, &identity, 10_000, &mut index)
        .await;
    assert!(
        result
            .expect_err("changed head blocks publication")
            .to_string()
            .contains("устаревший review")
    );
    let _ = deepseek_server
        .await
        .expect("DeepSeek fake handled the review request");
    control.send(MockGithubControl::Shutdown).unwrap();
    let github_state = github_server.await.unwrap();
    assert!(
        github_state.requests.iter().all(|request| {
            !request.path.ends_with("/pulls/7/comments")
                && !request.path.contains("/issues/7/comments")
                && !request.body["query"]
                    .as_str()
                    .is_some_and(|query| query.contains("resolveReviewThread"))
        }),
        "stale run performs no summary, inline-comment, or resolve write"
    );
    assert!(github_state.threads.is_empty());
    assert!(github_state.summaries.is_empty());
}

#[tokio::test]
async fn d32_review_resolution_requires_new_code_not_unchanged_context() {
    let fixture = review_project_fixture();
    let root = fixture.path();
    write_text(
        root,
        "src/transaction.rs",
        "pub fn calculate() {\n    if !validate_transaction_balance() {\n        return;\n    }\n    calculate_transaction_total();\n}\n",
    );
    let head_one = commit_fixture(root, "add transaction balance guard");
    let base_sha = "b".repeat(40);
    let patch_one = "@@ -0,0 +1,6 @@\n+pub fn calculate() {\n+    if !validate_transaction_balance() {\n+        return;\n+    }\n+    calculate_transaction_total();\n+}\n";
    let (github_url, github_server, control) = mock_github_server(
        base_sha.clone(),
        head_one.clone(),
        patch_one.to_owned(),
        None,
    )
    .await;
    let mut finding = review_finding();
    finding.line = Some(5);
    let fingerprint = finding_fingerprint(&finding);
    let responses = vec![
        deepseek_review_response(review_report(
            "Проверить порядок расчета и balance guard.",
            Some(&finding),
            Vec::new(),
        )),
        deepseek_review_response(review_report(
            "Изменена только поясняющая строка.",
            None,
            vec![json!({
                "fingerprint": fingerprint,
                "path": "src/transaction.rs",
                "evidence": "    if !validate_transaction_balance() {"
            })],
        )),
    ];
    let (deepseek_url, deepseek_server) = scripted_http(responses).await;
    let runner = ReviewRunner::new(
        GitHubClient::new(github_url, "synthetic-github-token".to_owned()).unwrap(),
        DeepSeekClient::new(
            deepseek_url,
            "synthetic-deepseek-key".to_owned(),
            "deepseek-flash".to_owned(),
        )
        .unwrap(),
    );
    let identity_one = PullRequestId {
        owner: "acme".to_owned(),
        repository: "transaction-import".to_owned(),
        number: 7,
        expected_base_sha: base_sha,
        expected_head_sha: head_one,
    };
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-context-resolution/index"),
        FixtureEmbedder::default(),
    )
    .expect("synthetic current head indexes");
    let first = runner
        .run_with_index(root, &identity_one, 10_000, &mut index)
        .await
        .expect("initial finding is published");
    assert_eq!(first.threads_created_or_updated, 1);

    write_text(
        root,
        "src/transaction.rs",
        "pub fn calculate() {\n    if !validate_transaction_balance() {\n        return;\n    }\n    // Keep the validation step visible.\n    calculate_transaction_total();\n}\n",
    );
    let head_two = commit_fixture(root, "clarify validation flow");
    let patch_two = "@@ -1,6 +1,7 @@\n pub fn calculate() {\n     if !validate_transaction_balance() {\n         return;\n     }\n+    // Keep the validation step visible.\n     calculate_transaction_total();\n }\n";
    set_mock_snapshot(&control, head_two.clone(), patch_two.to_owned()).await;
    drop(index);
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-context-resolution/index"),
        FixtureEmbedder::default(),
    )
    .expect("updated head rebuilds its index");
    let identity_two = PullRequestId {
        expected_head_sha: head_two,
        ..identity_one
    };
    let second = runner
        .run_with_index(root, &identity_two, 10_000, &mut index)
        .await
        .expect("context-only resolution suggestion is safely ignored");
    assert_eq!(second.threads_resolved, 0);

    let _ = deepseek_server
        .await
        .expect("DeepSeek fake answered both review requests");
    control.send(MockGithubControl::Shutdown).unwrap();
    let github_state = github_server.await.unwrap();
    assert_eq!(github_state.threads.len(), 1);
    assert!(!github_state.threads[0].is_resolved);
    assert!(github_state.requests.iter().all(|request| {
        !request.body["query"]
            .as_str()
            .is_some_and(|query| query.contains("resolveReviewThread"))
    }));
}

#[tokio::test]
async fn d32_review_google_api_keys_in_diff_never_reach_provider_or_summary() {
    let fixture = review_project_fixture();
    let root = fixture.path();
    let base_sha = "b".repeat(40);
    let head_sha = git_head(root);
    let secret = "AIza0123456789abcdefghijklmnopqrstuvwxyzAB";
    let patch = format!(
        "@@ -0,0 +1,2 @@\n+pub const GOOGLE_MAPS_API_KEY: &str = \"{secret}\";\n+pub const OTHER: &str = \"{secret}\";\n"
    );
    let (github_url, github_server, control) =
        mock_github_server(base_sha.clone(), head_sha.clone(), patch, None).await;
    let (deepseek_url, deepseek_server) =
        scripted_http(vec![deepseek_review_response(review_report(
            "Проверенная часть diff не содержит findings.",
            None,
            Vec::new(),
        ))])
        .await;
    let runner = ReviewRunner::new(
        GitHubClient::new(github_url, "synthetic-github-token".to_owned()).unwrap(),
        DeepSeekClient::new(
            deepseek_url,
            "synthetic-deepseek-key".to_owned(),
            "deepseek-flash".to_owned(),
        )
        .unwrap(),
    );
    let identity = PullRequestId {
        owner: "acme".to_owned(),
        repository: "transaction-import".to_owned(),
        number: 7,
        expected_base_sha: base_sha,
        expected_head_sha: head_sha,
    };
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-google-secret/index"),
        FixtureEmbedder::default(),
    )
    .expect("synthetic corpus indexes");

    let result = runner
        .run_with_index(root, &identity, 10_000, &mut index)
        .await
        .expect("filtered secret-bearing file does not block safe file review");
    let deepseek_requests = deepseek_server
        .await
        .expect("DeepSeek fake handled one safe review request");
    control.send(MockGithubControl::Shutdown).unwrap();
    let github_state = github_server.await.unwrap();

    assert_eq!(result.changed_files_reviewed, 1);
    let prompt = request_user_json(&deepseek_requests[0]);
    assert_eq!(prompt["changed_files"], json!(["src/fees.rs"]));
    assert!(
        prompt["diff"]
            .as_str()
            .unwrap()
            .contains("write_fee_audit_log")
    );
    assert!(!prompt.to_string().contains(secret));
    assert!(
        github_state
            .requests
            .iter()
            .all(|request| !request.body.to_string().contains(secret))
    );
    assert!(
        github_state
            .summaries
            .iter()
            .all(|summary| !summary.body.contains(secret))
    );
}

#[tokio::test]
async fn d32_review_secret_like_model_output_is_rejected_before_github_writes() {
    let fixture = review_project_fixture();
    let root = fixture.path();
    let base_sha = "b".repeat(40);
    let head_sha = git_head(root);
    let (github_url, github_server, control) = mock_github_server(
        base_sha.clone(),
        head_sha.clone(),
        "@@ -0,0 +1,3 @@\n+pub fn calculate() {\n+    calculate_transaction_total();\n+}\n"
            .to_owned(),
        None,
    )
    .await;
    let secret = "AIza0123456789abcdefghijklmnopqrstuvwxyzAB";
    let mut finding = review_finding();
    finding.detail = format!("Untrusted output contains {secret}.");
    let (deepseek_url, deepseek_server) = scripted_http(vec![deepseek_review_response(
        review_report("Проверка завершена.", Some(&finding), Vec::new()),
    )])
    .await;
    let runner = ReviewRunner::new(
        GitHubClient::new(github_url, "synthetic-github-token".to_owned()).unwrap(),
        DeepSeekClient::new(
            deepseek_url,
            "synthetic-deepseek-key".to_owned(),
            "deepseek-flash".to_owned(),
        )
        .unwrap(),
    );
    let identity = PullRequestId {
        owner: "acme".to_owned(),
        repository: "transaction-import".to_owned(),
        number: 7,
        expected_base_sha: base_sha,
        expected_head_sha: head_sha,
    };
    let mut index = ReviewIndex::open(
        root,
        &root.join(".data/review-secret-output/index"),
        FixtureEmbedder::default(),
    )
    .expect("synthetic corpus indexes");

    let error = runner
        .run_with_index(root, &identity, 10_000, &mut index)
        .await
        .expect_err("secret-like model output is rejected before publishing");
    let _ = deepseek_server
        .await
        .expect("DeepSeek fake handled one review request");
    control.send(MockGithubControl::Shutdown).unwrap();
    let github_state = github_server.await.unwrap();

    assert!(error.to_string().contains("небезопасное поле"));
    assert!(github_state.threads.is_empty());
    assert!(github_state.summaries.is_empty());
    assert!(
        github_state.requests.iter().all(|request| {
            !(request.method == "PATCH"
                || (request.method == "POST"
                    && (request.path.ends_with("/pulls/7/comments")
                        || request.path.ends_with("/issues/7/comments")
                        || request.body["query"]
                            .as_str()
                            .is_some_and(|query| query.contains("resolveReviewThread")))))
        }),
        "untrusted model output causes no GitHub write"
    );
    assert!(
        github_state
            .requests
            .iter()
            .all(|request| !request.body.to_string().contains(secret))
    );
}
#[tokio::test]
async fn d32_review_missing_usage_fails_closed_before_returning_model_content() {
    let responses = vec![json!({
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "{\"summary\":\"untrusted\",\"findings\":[],\"resolved_findings\":[]}"
            }
        }]
    })];
    let (endpoint, server) = scripted_http(responses).await;
    let client = DeepSeekClient::new(
        endpoint,
        "synthetic-review-key".to_owned(),
        "deepseek-flash".to_owned(),
    )
    .unwrap();
    let mut tools = CountingTools::default();
    let result = client
        .review("Only review.", "Synthetic diff.", &mut tools, 100)
        .await;
    assert!(
        result
            .expect_err("missing uncached token usage fails closed")
            .to_string()
            .contains("token usage")
    );
    assert_eq!(server.await.expect("fake provider answered once").len(), 1);
}

fn review_project_fixture() -> TempDir {
    let fixture = project_fixture();
    write_text(
        fixture.path(),
        "src/transaction.rs",
        "pub fn calculate() {\n    calculate_transaction_total();\n}\n",
    );
    write_text(
        fixture.path(),
        "src/fees.rs",
        "pub fn calculate_fee() {\n    calculate_transaction_total();\n    write_fee_audit_log();\n}\n",
    );
    let _ = commit_fixture(fixture.path(), "initial review fixture");
    fixture
}

fn commit_fixture(root: &Path, message: &str) -> String {
    let add = git(root, &["add", "-A"]);
    assert!(
        add.status.success(),
        "git add failed: {}",
        String::from_utf8_lossy(&add.stderr)
    );
    let commit = git(
        root,
        &[
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.invalid",
            "commit",
            "-m",
            message,
        ],
    );
    assert!(
        commit.status.success(),
        "git commit failed: {}",
        String::from_utf8_lossy(&commit.stderr)
    );
    git_head(root)
}

fn git_head(root: &Path) -> String {
    let output = git(root, &["rev-parse", "HEAD"]);
    assert!(
        output.status.success(),
        "git rev-parse failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("commit SHA is UTF-8")
        .trim()
        .to_owned()
}

fn review_finding() -> ReviewFinding {
    ReviewFinding {
        category: FindingCategory::Bug,
        title: "Missing balance validation".to_owned(),
        detail: "The total is calculated before the transaction balance is checked.".to_owned(),
        path: "src/transaction.rs".to_owned(),
        line: Some(2),
        evidence: "    calculate_transaction_total();".to_owned(),
    }
}

fn review_findings() -> Vec<ReviewFinding> {
    vec![
        review_finding(),
        ReviewFinding {
            category: FindingCategory::Architecture,
            title: "Fee calculation crosses boundaries".to_owned(),
            detail: "The module exposes a calculation before validating its inputs.".to_owned(),
            path: "src/fees.rs".to_owned(),
            line: Some(1),
            evidence: "pub fn calculate_fee() {".to_owned(),
        },
        ReviewFinding {
            category: FindingCategory::Recommendation,
            title: "Preserve fee auditability".to_owned(),
            detail: "Record a successful fee calculation for later diagnosis.".to_owned(),
            path: "src/fees.rs".to_owned(),
            line: Some(3),
            evidence: "    write_fee_audit_log();".to_owned(),
        },
    ]
}

fn review_report(summary: &str, finding: Option<&ReviewFinding>, resolved: Vec<Value>) -> Value {
    let findings = finding.into_iter().cloned().collect::<Vec<_>>();
    review_report_many(summary, &findings, resolved)
}

fn review_report_many(summary: &str, findings: &[ReviewFinding], resolved: Vec<Value>) -> Value {
    let findings = findings
        .iter()
        .map(|finding| {
            json!({
                "category": finding.category.as_str(),
                "title": finding.title,
                "detail": finding.detail,
                "path": finding.path,
                "line": finding.line,
                "evidence": finding.evidence
            })
        })
        .collect::<Vec<_>>();
    json!({
        "summary": summary,
        "findings": findings,
        "resolved_findings": resolved
    })
}

fn deepseek_review_response(report: Value) -> Value {
    deepseek_response(
        "stop",
        json!({
            "role": "assistant",
            "content": serde_json::to_string(&report).expect("report serializes")
        }),
        1_000,
        700,
        300,
    )
}

fn request_user_json(request: &CapturedRequest) -> Value {
    let content = request.body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "user")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    serde_json::from_str(content).expect("review user message is bounded JSON data")
}

#[derive(Debug)]
struct MockGithubThread {
    thread_id: String,
    comment_id: u64,
    body: String,
    path: String,
    line: usize,
    is_resolved: bool,
    is_outdated: bool,
}

#[derive(Debug)]
struct MockGithubSummary {
    comment_id: u64,
    body: String,
}

#[derive(Debug)]
struct MockGithubState {
    base_sha: String,
    head_sha: String,
    patch: String,
    stale_on_metadata_call: Option<usize>,
    metadata_calls: usize,
    next_comment_id: u64,
    threads: Vec<MockGithubThread>,
    summaries: Vec<MockGithubSummary>,
    requests: Vec<CapturedRequest>,
}

enum MockGithubControl {
    SetSnapshot {
        head_sha: String,
        patch: String,
        ack: oneshot::Sender<()>,
    },
    UpdateThread {
        comment_id: u64,
        body: String,
        is_outdated: bool,
        ack: oneshot::Sender<()>,
    },
    Shutdown,
}

async fn mock_github_server(
    base_sha: String,
    head_sha: String,
    patch: String,
    stale_on_metadata_call: Option<usize>,
) -> (
    Url,
    JoinHandle<MockGithubState>,
    tokio::sync::mpsc::UnboundedSender<MockGithubControl>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("fake GitHub endpoint binds");
    let address = listener.local_addr().expect("fake GitHub endpoint address");
    let endpoint =
        Url::parse(&format!("http://{address}/api/")).expect("fake GitHub API URL is valid");
    let (control, mut controls) = tokio::sync::mpsc::unbounded_channel();
    let server = tokio::spawn(async move {
        let mut state = MockGithubState {
            base_sha,
            head_sha,
            patch,
            stale_on_metadata_call,
            metadata_calls: 0,
            next_comment_id: 500,
            threads: Vec::new(),
            summaries: Vec::new(),
            requests: Vec::new(),
        };
        loop {
            let accepted = tokio::select! {
                control = controls.recv() => {
                    match control {
                        Some(MockGithubControl::SetSnapshot { head_sha, patch, ack }) => {
                            state.head_sha = head_sha;
                            state.patch = patch;
                            let _ = ack.send(());
                            continue;
                        }
                        Some(MockGithubControl::UpdateThread {
                            comment_id,
                            body,
                            is_outdated,
                            ack,
                        }) => {
                            let thread = state
                                .threads
                                .iter_mut()
                                .find(|thread| thread.comment_id == comment_id)
                                .expect("fake GitHub thread exists");
                            thread.body = body;
                            thread.is_outdated = is_outdated;
                            let _ = ack.send(());
                            continue;
                        }
                        Some(MockGithubControl::Shutdown) | None => break,
                    }
                }
                accepted = listener.accept() => accepted.expect("fake GitHub accepts a request"),
            };
            let (mut stream, _) = accepted;
            let request = read_http_request(&mut stream).await;
            let (status, response) = mock_github_response(&request, &mut state);
            state.requests.push(request);
            write_json_response(&mut stream, status, &response).await;
        }
        state
    });
    (endpoint, server, control)
}

async fn set_mock_snapshot(
    control: &tokio::sync::mpsc::UnboundedSender<MockGithubControl>,
    head_sha: String,
    patch: String,
) {
    let (ack, received) = oneshot::channel();
    control
        .send(MockGithubControl::SetSnapshot {
            head_sha,
            patch,
            ack,
        })
        .expect("fake GitHub accepts a snapshot update");
    received.await.expect("fake GitHub applied snapshot update");
}

async fn update_mock_thread(
    control: &tokio::sync::mpsc::UnboundedSender<MockGithubControl>,
    comment_id: u64,
    body: String,
    is_outdated: bool,
) {
    let (ack, received) = oneshot::channel();
    control
        .send(MockGithubControl::UpdateThread {
            comment_id,
            body,
            is_outdated,
            ack,
        })
        .expect("fake GitHub accepts a thread update");
    received.await.expect("fake GitHub applied thread update");
}

fn mock_github_response(request: &CapturedRequest, state: &mut MockGithubState) -> (u16, Value) {
    let path = request.path.split('?').next().unwrap();
    match (request.method.as_str(), path) {
        ("POST", "/api/graphql")
            if request.body["query"].as_str().is_some_and(|query| {
                query.contains("viewer") && !query.contains("reviewThreads")
            }) =>
        {
            (200, json!({"data":{"viewer":{"login":"review-bot"}}}))
        }
        ("GET", "/api/repos/acme/transaction-import/pulls/7") => {
            state.metadata_calls += 1;
            let head_sha = if state.stale_on_metadata_call == Some(state.metadata_calls) {
                "c".repeat(40)
            } else {
                state.head_sha.clone()
            };
            (
                200,
                json!({
                    "state": "open",
                    "base": {
                        "sha": state.base_sha,
                        "repo": {"full_name":"acme/transaction-import"}
                    },
                    "head": {
                        "sha": head_sha,
                        "repo": {"full_name":"acme/transaction-import"}
                    }
                }),
            )
        }
        ("GET", "/api/repos/acme/transaction-import/pulls/7/files") => (
            200,
            json!([
                {
                    "filename":"src/transaction.rs",
                    "status":"modified",
                    "patch":state.patch
                },
                {
                    "filename":"src/fees.rs",
                    "status":"modified",
                    "patch":"@@ -0,0 +1,4 @@\n+pub fn calculate_fee() {\n+    calculate_transaction_total();\n+    write_fee_audit_log();\n+}\n"
                },
                {
                    "filename":"assets/logo.png",
                    "status":"removed",
                    "patch":null
                }
            ]),
        ),
        ("GET", "/api/repos/acme/transaction-import/issues/7/comments") => (
            200,
            Value::Array(
                state
                    .summaries
                    .iter()
                    .map(|summary| {
                        json!({
                            "id":summary.comment_id,
                            "body":summary.body,
                            "user":{"login":"review-bot"}
                        })
                    })
                    .collect(),
            ),
        ),
        ("POST", "/api/graphql")
            if request.body["query"]
                .as_str()
                .is_some_and(|query| query.contains("resolveReviewThread")) =>
        {
            let thread_id = request.body["variables"]["threadId"].as_str().unwrap();
            let mut resolved = false;
            for thread in &mut state.threads {
                if thread.thread_id == thread_id {
                    thread.is_resolved = true;
                    resolved = true;
                }
            }
            (
                200,
                json!({
                    "data": {
                        "resolveReviewThread": {
                            "thread": {"id":thread_id,"isResolved":resolved}
                        }
                    }
                }),
            )
        }
        ("POST", "/api/graphql") => (
            200,
            json!({
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": state.threads.iter().map(|thread| json!({
                                    "id":thread.thread_id,
                                    "isResolved":thread.is_resolved,
                                    "isOutdated":thread.is_outdated,
                                    "path":thread.path,
                                    "line":thread.line,
                                    "comments":{"nodes":[{
                                        "databaseId":thread.comment_id,
                                        "body":thread.body,
                                        "author":{"login":"review-bot"}
                                    }]}
                                })).collect::<Vec<_>>(),
                                "pageInfo":{"hasNextPage":false,"endCursor":null}
                            }
                        }
                    }
                }
            }),
        ),
        ("POST", "/api/repos/acme/transaction-import/pulls/7/comments") => {
            let comment_id = state.next_comment_id;
            state.next_comment_id += 1;
            let thread_id = format!("PRRT_{comment_id}");
            state.threads.push(MockGithubThread {
                thread_id,
                comment_id,
                body: request.body["body"].as_str().unwrap().to_owned(),
                path: request.body["path"].as_str().unwrap().to_owned(),
                line: request.body["line"].as_u64().unwrap() as usize,
                is_resolved: false,
                is_outdated: false,
            });
            (200, json!({"id":comment_id}))
        }
        ("PATCH", path)
            if path.starts_with("/api/repos/acme/transaction-import/pulls/comments/") =>
        {
            let comment_id = path.rsplit('/').next().unwrap().parse::<u64>().unwrap();
            let body = request.body["body"].as_str().unwrap().to_owned();
            if let Some(thread) = state
                .threads
                .iter_mut()
                .find(|thread| thread.comment_id == comment_id)
            {
                thread.body = body;
                (200, json!({"id":comment_id}))
            } else {
                (404, json!({"message":"thread not found"}))
            }
        }
        ("POST", "/api/repos/acme/transaction-import/issues/7/comments") => {
            let comment_id = state.next_comment_id;
            state.next_comment_id += 1;
            state.summaries.push(MockGithubSummary {
                comment_id,
                body: request.body["body"].as_str().unwrap().to_owned(),
            });
            (200, json!({"id":comment_id}))
        }
        ("PATCH", path)
            if path.starts_with("/api/repos/acme/transaction-import/issues/comments/") =>
        {
            let comment_id = path.rsplit('/').next().unwrap().parse::<u64>().unwrap();
            let body = request.body["body"].as_str().unwrap().to_owned();
            if let Some(summary) = state
                .summaries
                .iter_mut()
                .find(|summary| summary.comment_id == comment_id)
            {
                summary.body = body;
                (200, json!({"id":comment_id}))
            } else {
                (404, json!({"message":"summary not found"}))
            }
        }
        _ => (404, json!({"message":"unexpected fake GitHub request"})),
    }
}

async fn write_json_response(stream: &mut TcpStream, status: u16, body: &Value) {
    let body = serde_json::to_vec(body).expect("fake GitHub response serializes");
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let mut header = format!("HTTP/1.1 {status} {reason}").into_bytes();
    header.extend_from_slice(&[13, 10]);
    header.extend_from_slice(b"Content-Type: application/json");
    header.extend_from_slice(&[13, 10]);
    header.extend_from_slice(format!("Content-Length: {}", body.len()).as_bytes());
    header.extend_from_slice(&[13, 10]);
    header.extend_from_slice(b"Connection: close");
    header.extend_from_slice(&[13, 10, 13, 10]);
    stream
        .write_all(&header)
        .await
        .expect("fake GitHub headers write");
    stream
        .write_all(&body)
        .await
        .expect("fake GitHub body writes");
}

#[test]
fn d32_review_cache_rejects_tracked_payloads_before_loading_model() {
    let fixture = project_fixture();
    let root = fixture.path();
    let tracked_path = ".data/developer-assistant-review/model/attacker.onnx";
    write_text(root, tracked_path, "synthetic untrusted cache payload\n");
    let add = git(root, &["add", "-f", tracked_path]);
    assert!(add.status.success(), "tracked cache fixture is staged");

    let cache = root.join(".data/developer-assistant-review");
    let index_result = ReviewIndex::open(root, &cache, OrthogonalEmbedder);
    let model_result = FastEmbedder::load(root, &cache);
    assert!(index_result.is_err());
    assert!(model_result.is_err());
}
