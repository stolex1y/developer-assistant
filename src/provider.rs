use anyhow::{Context, Result, bail};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{collections::HashSet, sync::LazyLock, time::Duration};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_COMPLETION_TOKENS: u32 = 4096;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOOL_ARGUMENT_BYTES: usize = 16 * 1024;
pub(crate) const MAX_TOOL_RESULT_BYTES: usize = 64 * 1024;

static HELP_TOOLS: LazyLock<Vec<Value>> = LazyLock::new(|| {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search relevant passages in the filtered project documentation and source corpus. Returns at most eight bounded passages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8}
                    },
                    "required": ["query", "limit"],
                    "additionalProperties": false
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read an exact line range from a file in the filtered project corpus. Reads at most 80 lines and 8 KiB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 1024},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1}
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": false
                }
            }
        }),
    ]
});

static TASK_TOOLS: LazyLock<Vec<Value>> = LazyLock::new(|| {
    let definitions = [
        (
            "list_files",
            "List allowed project paths; follow next_offset while truncated.",
            json!({"offset": {"type": "integer", "minimum": 0}}),
            Vec::<&str>::new(),
        ),
        (
            "search_text",
            "Literal search across allowed project files; results may be truncated.",
            json!({"query": {"type": "string", "minLength": 1}}),
            vec!["query"],
        ),
        (
            "read_file",
            "Read at most 200 lines and 32768 JSON bytes; use smaller ranges if needed. Read every line before proposing an existing file.",
            json!({"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}),
            vec!["path", "start_line", "end_line"],
        ),
        (
            "propose_file",
            "Stage complete UTF-8 content, without writing to disk.",
            json!({"path": {"type": "string"}, "content": {"type": "string"}}),
            vec!["path", "content"],
        ),
    ];
    definitions
        .into_iter()
        .map(|(name, description, properties, required)| {
            json!({
                "type": "function",
                "function": {"name": name, "description": description,
                    "parameters": {"type": "object", "properties": properties,
                        "required": required, "additionalProperties": false}}
            })
        })
        .collect()
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputTokenUsage {
    pub prompt_tokens: u64,
    pub cached_input_tokens: u64,
    pub uncached_input_tokens: u64,
}

pub(crate) fn parse_input_usage(
    usage: Option<Value>,
    provider_name: &str,
) -> Result<InputTokenUsage> {
    let usage = usage.with_context(|| {
        format!("{provider_name} не вернул token usage; tool loop остановлен без оценки.")
    })?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .with_context(|| format!("{provider_name} не вернул корректный prompt_tokens."))?;
    let details = usage
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .with_context(|| format!("{provider_name} не вернул prompt_tokens_details."))?;
    let prompt_cache_hit_tokens =
        optional_counter(details, "prompt_cache_hit_tokens", provider_name)?;
    let cached_tokens = optional_counter(details, "cached_tokens", provider_name)?;
    if matches!((prompt_cache_hit_tokens, cached_tokens), (Some(hit), Some(alias)) if hit != alias)
    {
        bail!("{provider_name} вернул противоречивые cache-hit token counts.");
    }
    let cache_hit_tokens = prompt_cache_hit_tokens.or(cached_tokens);
    let prompt_cache_miss_tokens =
        optional_counter(details, "prompt_cache_miss_tokens", provider_name)?;
    let (cache_hit_tokens, cache_miss_tokens) = match (cache_hit_tokens, prompt_cache_miss_tokens) {
        (Some(hit), Some(miss)) => (hit, miss),
        (Some(hit), None) => (
            hit,
            prompt_tokens.checked_sub(hit).with_context(|| {
                format!("{provider_name} вернул cache-hit больше общего prompt_tokens.")
            })?,
        ),
        (None, Some(miss)) => (
            prompt_tokens.checked_sub(miss).with_context(|| {
                format!("{provider_name} вернул uncached больше общего prompt_tokens.")
            })?,
            miss,
        ),
        (None, None) => {
            bail!(
                "{provider_name} не вернул cache-hit/cache-miss breakdown; tool loop остановлен."
            );
        }
    };
    if cache_hit_tokens.checked_add(cache_miss_tokens) != Some(prompt_tokens) {
        bail!("{provider_name} вернул противоречивый prompt token usage.");
    }
    Ok(InputTokenUsage {
        prompt_tokens,
        cached_input_tokens: cache_hit_tokens,
        uncached_input_tokens: cache_miss_tokens,
    })
}

fn optional_counter(
    details: &Map<String, Value>,
    name: &str,
    provider_name: &str,
) -> Result<Option<u64>> {
    match details.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .with_context(|| format!("{provider_name} вернул некорректный {name}.")),
    }
}

pub trait ToolExecutor {
    /// Returns the complete content for the tool-role message; serialize structured results once.
    fn execute(&mut self, name: &str, arguments: Value) -> Result<String>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolLoopResult {
    pub content: String,
    pub finish_reason: Option<String>,
    pub uncached_input_tokens: u64,
    pub tool_calls_executed: u64,
    pub requests: u64,
    pub final_usage: Option<InputTokenUsage>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "role")]
enum ChatMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'static [Value]>,
    tool_choice: &'static str,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantResponse,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct AssistantResponse {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

pub async fn tool_loop(
    config: &crate::config::ProviderConfig,
    system_prompt: &str,
    user_prompt: &str,
    tools: &mut impl ToolExecutor,
    max_uncached_input_tokens: u64,
) -> Result<ToolLoopResult> {
    tool_loop_with_schema(
        config,
        system_prompt,
        user_prompt,
        tools,
        max_uncached_input_tokens,
        HELP_TOOLS.as_slice(),
        false,
    )
    .await
}

pub async fn task_tool_loop(
    config: &crate::config::ProviderConfig,
    system_prompt: &str,
    user_prompt: &str,
    tools: &mut impl ToolExecutor,
) -> Result<ToolLoopResult> {
    tool_loop_with_schema(
        config,
        system_prompt,
        user_prompt,
        tools,
        30_000,
        TASK_TOOLS.as_slice(),
        true,
    )
    .await
}

async fn tool_loop_with_schema(
    config: &crate::config::ProviderConfig,
    system_prompt: &str,
    user_prompt: &str,
    tools: &mut impl ToolExecutor,
    max_uncached_input_tokens: u64,
    schema: &'static [Value],
    strict: bool,
) -> Result<ToolLoopResult> {
    if max_uncached_input_tokens == 0 {
        bail!("Лимит uncached input tokens должен быть больше нуля.");
    }
    let http = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|_| anyhow::anyhow!("Не удалось настроить HTTP-клиент provider."))?;
    let mut messages = vec![
        ChatMessage::System {
            content: system_prompt.to_owned(),
        },
        ChatMessage::User {
            content: user_prompt.to_owned(),
        },
    ];
    let mut seen_tool_calls = HashSet::new();
    let mut uncached_input_tokens = 0_u64;
    let mut tool_calls_executed = 0_u64;
    let mut requests = 0_u64;

    loop {
        let completion = request(config, &http, &messages, true, schema).await?;
        requests = requests.saturating_add(1);
        let usage = parse_input_usage(completion.usage, "Provider")?;
        uncached_input_tokens = uncached_input_tokens
            .checked_add(usage.uncached_input_tokens)
            .context("Сумма uncached input tokens переполнена.")?;
        let choice = first_choice(completion.choices)?;
        if strict && !matches!(choice.finish_reason.as_deref(), Some("stop" | "tool_calls")) {
            bail!("Неполный или некорректный ответ /task; изменения отменены.");
        }
        let tool_calls = choice.message.tool_calls;
        if tool_calls.is_empty() {
            let content = choice
                .message
                .content
                .filter(|content| !content.trim().is_empty())
                .context("Provider вернул пустой ответ.")?;
            return Ok(ToolLoopResult {
                content,
                finish_reason: choice.finish_reason,
                uncached_input_tokens,
                tool_calls_executed,
                requests,
                final_usage: None,
            });
        }
        validate_tool_calls(&tool_calls)?;
        messages.push(ChatMessage::Assistant {
            content: choice.message.content,
            tool_calls: tool_calls.clone(),
        });

        if uncached_input_tokens >= max_uncached_input_tokens {
            for call in tool_calls {
                messages.push(tool_result(
                    call.id,
                    json!({
                        "error": "input_budget_exhausted",
                        "message": "Uncached input-token budget reached; this tool was not executed."
                    }),
                )?);
            }
            let completion = request(config, &http, &messages, false, schema).await?;
            requests = requests.saturating_add(1);
            let final_usage = parse_input_usage(completion.usage, "Provider")?;
            let choice = first_choice(completion.choices)?;
            if strict && choice.finish_reason.as_deref() != Some("stop") {
                bail!("Неполный финальный ответ /task; изменения отменены.");
            }
            if !choice.message.tool_calls.is_empty() {
                bail!("Provider вернул tool call в финальном запросе без tools.");
            }
            let content = choice
                .message
                .content
                .filter(|content| !content.trim().is_empty())
                .context("Provider не вернул финальный ответ после исчерпания token budget.")?;
            return Ok(ToolLoopResult {
                content,
                finish_reason: choice.finish_reason,
                uncached_input_tokens,
                tool_calls_executed,
                requests,
                final_usage: Some(final_usage),
            });
        }

        for call in tool_calls {
            let signature = format!("{}\0{}", call.function.name, call.function.arguments);
            let result = if !seen_tool_calls.insert(signature) {
                Err(json!({
                    "error": "duplicate_tool_call",
                    "message": "This exact tool call was already attempted; use new evidence or provide the final answer."
                }))
            } else if call.function.arguments.len()
                > if strict {
                    MAX_RESPONSE_BYTES
                } else {
                    MAX_TOOL_ARGUMENT_BYTES
                }
            {
                Err(json!({
                    "error": "tool_arguments_too_large",
                    "message": "Tool arguments exceeded the size limit and were not executed."
                }))
            } else {
                match serde_json::from_str::<Value>(&call.function.arguments) {
                    Ok(arguments) => match tools.execute(&call.function.name, arguments) {
                        Ok(result) => {
                            tool_calls_executed = tool_calls_executed.saturating_add(1);
                            if strict && call.function.name == "propose_file" {
                                // Discovery now observes a new staging view, not the old result.
                                seen_tool_calls
                                    .retain(|signature| signature.starts_with("propose_file\0"));
                            }
                            Ok(result)
                        }
                        Err(_) => Err(json!({
                            "error": "tool_execution_failed",
                            "message": "The requested read-only retrieval operation could not be completed."
                        })),
                    },
                    Err(_) => Err(json!({
                        "error": "invalid_tool_arguments",
                        "message": "Tool arguments were not valid JSON; provide valid arguments or continue with the answer."
                    })),
                }
            };
            if strict
                && (result.is_err()
                    || result
                        .as_ref()
                        .is_ok_and(|content| content.len() > MAX_TOOL_RESULT_BYTES))
            {
                bail!("Ошибка инструмента /task; изменения отменены.");
            }
            let message = match result {
                Ok(content) => tool_result_content(call.id, content),
                Err(error) => tool_result(call.id, error)?,
            };
            messages.push(message);
        }
    }
}

async fn request(
    config: &crate::config::ProviderConfig,
    http: &Client,
    messages: &[ChatMessage],
    tools_enabled: bool,
    schema: &'static [Value],
) -> Result<ChatCompletionResponse> {
    let request = ChatCompletionRequest {
        model: config.model(),
        messages,
        temperature: 0.0,
        max_tokens: MAX_COMPLETION_TOKENS,
        tools: tools_enabled.then_some(schema),
        tool_choice: if tools_enabled { "auto" } else { "none" },
    };
    let response = http
        .post(config.endpoint().clone())
        .bearer_auth(config.api_key())
        .json(&request)
        .send()
        .await
        .map_err(|_| anyhow::anyhow!("Не удалось подключиться к provider."))?;
    let status = response.status();
    if !status.is_success() {
        bail!("Provider вернул HTTP {status}.");
    }
    let body = bounded_body(response).await?;
    serde_json::from_slice(&body).context("Provider вернул некорректный JSON.")
}

fn first_choice(choices: Vec<Choice>) -> Result<Choice> {
    choices
        .into_iter()
        .next()
        .context("Provider не вернул completion choice.")
}

fn validate_tool_calls(calls: &[ToolCall]) -> Result<()> {
    let mut ids = HashSet::with_capacity(calls.len());
    for call in calls {
        if call.call_type != "function"
            || call.id.is_empty()
            || call.id.len() > 256
            || call.id.chars().any(char::is_control)
            || !ids.insert(call.id.as_str())
        {
            bail!("Provider вернул некорректные tool-call IDs или type.");
        }
    }
    Ok(())
}

fn tool_result(tool_call_id: String, result: Value) -> Result<ChatMessage> {
    let content =
        serde_json::to_string(&result).context("Не удалось сериализовать tool result.")?;
    Ok(tool_result_content(tool_call_id, content))
}

fn tool_result_content(tool_call_id: String, content: String) -> ChatMessage {
    ChatMessage::Tool {
        tool_call_id,
        content: bounded_tool_result(content),
    }
}

fn bounded_tool_result(result: String) -> String {
    if result.len() <= MAX_TOOL_RESULT_BYTES {
        result
    } else {
        json!({
            "error": "tool_result_too_large",
            "message": "Tool output exceeded the response limit and was not returned."
        })
        .to_string()
    }
}

async fn bounded_body(mut response: Response) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        bail!("Ответ provider превышает допустимый размер.");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| anyhow::anyhow!("Не удалось прочитать ответ provider."))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            bail!("Ответ provider превышает допустимый размер.");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    #[test]
    fn task_schema_does_not_expand_help_write_boundary() {
        let names = |tools: &[serde_json::Value]| {
            tools
                .iter()
                .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(super::HELP_TOOLS.as_slice()),
            ["search_code", "read_file"]
        );
        assert_eq!(
            names(super::TASK_TOOLS.as_slice()),
            ["list_files", "search_text", "read_file", "propose_file"]
        );
    }

    use super::{InputTokenUsage, ToolExecutor, parse_input_usage, tool_loop};
    use crate::config::ProviderConfig;
    use anyhow::Result;
    use serde_json::{Value, json};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        task::JoinHandle,
    };

    #[test]
    fn uncached_usage_accepts_exact_hit_and_miss_counts() {
        let usage = parse_input_usage(
            Some(json!({
                "prompt_tokens": 9,
                "prompt_tokens_details": {
                    "prompt_cache_hit_tokens": 5,
                    "prompt_cache_miss_tokens": 4
                }
            })),
            "Provider",
        )
        .unwrap();
        assert_eq!(
            usage,
            InputTokenUsage {
                prompt_tokens: 9,
                cached_input_tokens: 5,
                uncached_input_tokens: 4
            }
        );
    }

    #[test]
    fn uncached_usage_accepts_cached_alias_and_derives_missing_counter() {
        let usage = parse_input_usage(
            Some(json!({
                "prompt_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 5}
            })),
            "Provider",
        )
        .unwrap();
        assert_eq!(usage.cached_input_tokens, 5);
        assert_eq!(usage.uncached_input_tokens, 4);
    }

    #[test]
    fn uncached_usage_rejects_missing_or_contradictory_counters() {
        assert!(parse_input_usage(None, "Provider").is_err());
        assert!(
            parse_input_usage(
                Some(json!({"prompt_tokens": 9, "prompt_tokens_details": {}})),
                "Provider"
            )
            .is_err()
        );
        assert!(
            parse_input_usage(
                Some(json!({
                    "prompt_tokens": 9,
                    "prompt_tokens_details": {
                        "prompt_cache_hit_tokens": 4,
                        "cached_tokens": 5,
                        "prompt_cache_miss_tokens": 4
                    }
                })),
                "Provider"
            )
            .is_err()
        );
        assert!(
            parse_input_usage(
                Some(json!({
                    "prompt_tokens": 9,
                    "prompt_tokens_details": {"prompt_cache_miss_tokens": 10}
                })),
                "Provider"
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn zero_uncached_tool_turns_are_allowed_below_the_budget() {
        let cached_turn = tool_call_response(Some(json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 100,
                "prompt_cache_miss_tokens": 0
            }
        })));
        let (config, server) = start_fake_provider(vec![
            cached_turn,
            completion_response(
                "answer",
                json!({
                    "prompt_tokens": 4,
                    "prompt_tokens_details": {
                        "prompt_cache_hit_tokens": 1,
                        "prompt_cache_miss_tokens": 3
                    }
                }),
            ),
        ])
        .await;
        let mut tools = CountingExecutor::default();

        let result = tool_loop(&config, "system", "question", &mut tools, 30_000)
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(result.uncached_input_tokens, 3);
        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(tools.calls, 1);
        assert_eq!(result.content, "answer");
    }

    #[tokio::test]
    async fn invalid_usage_rejects_pending_tools_before_execution() {
        let contradictory = json!({
            "prompt_tokens": 10,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 9,
                "prompt_cache_miss_tokens": 2
            }
        });
        for usage in [None, Some(contradictory)] {
            let (config, server) = start_fake_provider(vec![tool_call_response(usage)]).await;
            let mut tools = CountingExecutor::default();

            assert!(
                tool_loop(&config, "system", "question", &mut tools, 30_000)
                    .await
                    .is_err()
            );
            server.await.unwrap();
            assert_eq!(tools.calls, 0);
        }
    }

    #[derive(Default)]
    struct CountingExecutor {
        calls: usize,
    }

    impl ToolExecutor for CountingExecutor {
        fn execute(&mut self, _name: &str, _arguments: Value) -> Result<String> {
            self.calls += 1;
            Ok("tool result".to_owned())
        }
    }

    fn tool_call_response(usage: Option<Value>) -> Value {
        let mut response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": "{\"query\":\"marker\",\"limit\":1}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        if let Some(usage) = usage {
            response["usage"] = usage;
        }
        response
    }

    fn completion_response(content: &str, usage: Value) -> Value {
        json!({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": usage
        })
    }

    async fn start_fake_provider(responses: Vec<Value>) -> (ProviderConfig, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let endpoint = format!("http://{address}/v1/chat/completions");
        let server = tokio::spawn(async move {
            for body in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_http_request(&mut stream).await;
                let body = body.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        (ProviderConfig::for_test(&endpoint), server)
    }

    async fn read_http_request(stream: &mut TcpStream) {
        let mut request = Vec::new();
        loop {
            let mut buffer = [0; 2_048];
            let read = stream.read(&mut buffer).await.unwrap();
            assert_ne!(read, 0, "provider request ended before its body");
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
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
                .expect("HTTP provider request has Content-Length");
            if request.len() >= header_end + 4 + content_length {
                return;
            }
        }
    }
}
