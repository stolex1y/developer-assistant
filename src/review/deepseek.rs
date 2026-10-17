use anyhow::{Context, Result, bail};
use reqwest::{Client, Response, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, net::IpAddr, str::FromStr, time::Duration};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_COMPLETION_TOKENS: u32 = 4096;
const MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;

static TOOLS: std::sync::LazyLock<Vec<Value>> = std::sync::LazyLock::new(|| {
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

pub trait ReviewToolExecutor {
    fn execute(&mut self, name: &str, arguments: Value) -> Result<String>;
}

pub struct DeepSeekClient {
    endpoint: Url,
    api_key: String,
    model: String,
    http: Client,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolLoopResult {
    pub content: String,
    pub retrieval_uncached_input_tokens: u64,
    pub final_usage: Option<InputTokenUsage>,
    pub tool_calls_executed: u64,
    pub requests: u64,
}

pub use crate::provider::InputTokenUsage;

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    max_tokens: u32,
    thinking: ThinkingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'static [Value]>,
    tool_choice: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ThinkingMode {
    r#type: &'static str,
}

#[derive(Serialize)]
struct ResponseFormat {
    r#type: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct Choice {
    finish_reason: String,
    message: AssistantResponse,
}

#[derive(Deserialize)]
struct AssistantResponse {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

impl DeepSeekClient {
    pub fn new(endpoint: Url, api_key: String, model: String) -> Result<Self> {
        validate_endpoint(&endpoint)?;
        if api_key.trim().is_empty() || api_key.chars().any(char::is_control) {
            bail!("DeepSeek API key отсутствует или содержит управляющие символы.");
        }
        if model.trim().is_empty() || model.chars().any(char::is_control) {
            bail!("DeepSeek model ID некорректен.");
        }

        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|_| anyhow::anyhow!("Не удалось настроить HTTP-клиент DeepSeek."))?;
        Ok(Self {
            endpoint,
            api_key,
            model,
            http,
        })
    }

    pub async fn review(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tools: &mut impl ReviewToolExecutor,
        max_uncached_input_tokens: u64,
    ) -> Result<ToolLoopResult> {
        if max_uncached_input_tokens == 0 {
            bail!("Лимит uncached input tokens должен быть больше нуля.");
        }

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
        let mut final_usage = None;
        let mut tool_calls_executed = 0_u64;
        let mut requests = 0_u64;

        loop {
            let completion = self.request(&messages, true).await?;
            requests = requests.saturating_add(1);
            let choice = first_choice(completion.choices)?;
            let usage = valid_usage(completion.usage)?;
            uncached_input_tokens = uncached_input_tokens
                .checked_add(usage.uncached_input_tokens)
                .context("Сумма uncached input tokens переполнена.")?;
            let tool_calls = choice.message.tool_calls;
            if !tool_calls.is_empty() && usage.uncached_input_tokens == 0 {
                bail!(
                    "DeepSeek запросил инструменты без роста uncached input token budget; цикл остановлен."
                );
            }
            if tool_calls.is_empty() {
                let content = choice
                    .message
                    .content
                    .filter(|content| !content.trim().is_empty())
                    .context("DeepSeek не вернул текстовый review.")?;
                return Ok(ToolLoopResult {
                    content,
                    retrieval_uncached_input_tokens: uncached_input_tokens,
                    final_usage,
                    tool_calls_executed,
                    requests,
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
                let completion = self.request(&messages, false).await?;
                requests = requests.saturating_add(1);
                let usage = valid_usage(completion.usage)?;
                final_usage = Some(usage);
                let choice = first_choice(completion.choices)?;
                if !choice.message.tool_calls.is_empty() {
                    bail!("DeepSeek вернул tool call в финальном запросе без tools.");
                }
                let content = choice
                    .message
                    .content
                    .filter(|content| !content.trim().is_empty())
                    .context(
                        "DeepSeek не вернул финальный review после исчерпания token budget.",
                    )?;
                return Ok(ToolLoopResult {
                    content,
                    retrieval_uncached_input_tokens: uncached_input_tokens,
                    final_usage,
                    tool_calls_executed,
                    requests,
                });
            }

            for call in tool_calls {
                let signature = format!("{}\0{}", call.function.name, call.function.arguments);
                let result = if !seen_tool_calls.insert(signature) {
                    json!({
                        "error": "duplicate_tool_call",
                        "message": "This exact tool call was already attempted; use new evidence or provide the final review."
                    })
                    .to_string()
                } else {
                    match serde_json::from_str::<Value>(&call.function.arguments) {
                        Ok(arguments) => match tools.execute(&call.function.name, arguments) {
                            Ok(result) => {
                                tool_calls_executed = tool_calls_executed.saturating_add(1);
                                bounded_tool_result(result)
                            }
                            Err(_) => json!({
                                "error": "tool_execution_failed",
                                "message": "The requested read-only retrieval operation could not be completed."
                            })
                            .to_string(),
                        },
                        Err(_) => json!({
                            "error": "invalid_tool_arguments",
                            "message": "Tool arguments were not valid JSON; provide valid arguments or continue with the review."
                        })
                        .to_string(),
                    }
                };
                messages.push(ChatMessage::Tool {
                    tool_call_id: call.id,
                    content: result,
                });
            }
        }
    }

    async fn request(
        &self,
        messages: &[ChatMessage],
        tools_enabled: bool,
    ) -> Result<ChatCompletionResponse> {
        let request = ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature: 0.0,
            max_tokens: MAX_COMPLETION_TOKENS,
            thinking: ThinkingMode { r#type: "disabled" },
            tools: tools_enabled.then_some(TOOLS.as_slice()),
            tool_choice: if tools_enabled { "auto" } else { "none" },
            response_format: (!tools_enabled).then_some(ResponseFormat {
                r#type: "json_object",
            }),
        };
        let response = self
            .http
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("Не удалось подключиться к DeepSeek."))?;
        if !response.status().is_success() {
            bail!("DeepSeek вернул HTTP {}.", response.status());
        }
        let body = bounded_body(response, MAX_RESPONSE_BYTES).await?;
        serde_json::from_slice(&body).context("DeepSeek вернул некорректный JSON или usage.")
    }
}

fn validate_endpoint(endpoint: &Url) -> Result<()> {
    if !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        bail!("DeepSeek endpoint не должен содержать credentials, query или fragment.");
    }
    let host = endpoint
        .host_str()
        .context("DeepSeek endpoint должен содержать host.")?;
    let loopback = host.eq_ignore_ascii_case("localhost")
        || IpAddr::from_str(host.trim_matches(['[', ']']))
            .is_ok_and(|address| address.is_loopback());
    if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && loopback) {
        bail!("Удалённый DeepSeek endpoint должен использовать HTTPS.");
    }
    Ok(())
}

fn valid_usage(usage: Option<Value>) -> Result<InputTokenUsage> {
    crate::provider::parse_input_usage(usage, "DeepSeek")
}

fn first_choice(choices: Vec<Choice>) -> Result<Choice> {
    let choice = choices
        .into_iter()
        .next()
        .context("DeepSeek не вернул completion choice.")?;
    if choice.finish_reason == "length" {
        bail!(
            "DeepSeek завершил ответ с finish_reason=length; completion неполный, review остановлен."
        );
    }
    if choice.finish_reason == "content_filter"
        || choice.finish_reason == "insufficient_system_resource"
    {
        bail!("DeepSeek завершил review без usable completion.");
    }
    Ok(choice)
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
            bail!("DeepSeek вернул некорректные tool-call IDs или type.");
        }
    }
    Ok(())
}

fn tool_result(tool_call_id: String, value: Value) -> Result<ChatMessage> {
    let content = serde_json::to_string(&value).context("Не удалось сериализовать tool result.")?;
    Ok(ChatMessage::Tool {
        tool_call_id,
        content: bounded_tool_result(content),
    })
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

async fn bounded_body(mut response: Response, limit: usize) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        bail!("Ответ DeepSeek превышает допустимый размер.");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| anyhow::anyhow!("Не удалось прочитать ответ DeepSeek."))?
    {
        if body.len().saturating_add(chunk.len()) > limit {
            bail!("Ответ DeepSeek превышает допустимый размер.");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::{InputTokenUsage, valid_usage};
    use serde_json::json;

    #[test]
    fn token_usage_requires_consistent_uncached_count() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 4
            }
        }))
        .unwrap();
        assert_eq!(
            valid_usage(Some(usage)).unwrap(),
            InputTokenUsage {
                prompt_tokens: 9,
                cached_input_tokens: 5,
                uncached_input_tokens: 4
            }
        );
    }
    #[test]
    fn token_usage_derives_cache_hits_when_deepseek_omits_hit_count() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "prompt_cache_miss_tokens": 4
            }
        }))
        .unwrap();
        assert_eq!(
            valid_usage(Some(usage)).unwrap(),
            InputTokenUsage {
                prompt_tokens: 9,
                cached_input_tokens: 5,
                uncached_input_tokens: 4
            }
        );
    }

    #[test]
    fn token_usage_accepts_cached_tokens_alias() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "cached_tokens": 5,
                "prompt_cache_miss_tokens": 4
            }
        }))
        .unwrap();
        assert_eq!(
            valid_usage(Some(usage)).unwrap(),
            InputTokenUsage {
                prompt_tokens: 9,
                cached_input_tokens: 5,
                uncached_input_tokens: 4
            }
        );
    }

    #[test]
    fn token_usage_derives_uncached_count_when_miss_is_omitted() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "cached_tokens": 5
            }
        }))
        .unwrap();
        assert_eq!(
            valid_usage(Some(usage)).unwrap(),
            InputTokenUsage {
                prompt_tokens: 9,
                cached_input_tokens: 5,
                uncached_input_tokens: 4
            }
        );
    }

    #[test]
    fn token_usage_rejects_cache_hit_above_prompt_total() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "cached_tokens": 10
            }
        }))
        .unwrap();
        assert!(valid_usage(Some(usage)).is_err());
    }
    #[test]
    fn token_usage_rejects_conflicting_cache_hit_fields() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "prompt_cache_hit_tokens": 4,
                "cached_tokens": 5,
                "prompt_cache_miss_tokens": 4
            }
        }))
        .unwrap();
        assert!(valid_usage(Some(usage)).is_err());
    }

    #[test]
    fn token_usage_rejects_uncached_count_above_prompt_total() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {
                "prompt_cache_miss_tokens": 10
            }
        }))
        .unwrap();
        assert!(valid_usage(Some(usage)).is_err());
    }

    #[test]
    fn token_usage_requires_cache_breakdown() {
        let usage = serde_json::from_value(json!({
            "prompt_tokens": 9,
            "prompt_tokens_details": {}
        }))
        .unwrap();
        assert!(valid_usage(Some(usage)).is_err());
    }
}
