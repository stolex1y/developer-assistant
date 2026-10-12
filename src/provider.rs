use crate::config::ProviderConfig;
use anyhow::{Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_COMPLETION_TOKENS: u32 = 800;

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: [PromptMessage<'a>; 2],
    temperature: f32,
    max_completion_tokens: u32,
}

#[derive(Serialize)]
struct PromptMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: Option<String>,
}

pub async fn complete(
    config: &ProviderConfig,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<String> {
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|_| anyhow::anyhow!("Не удалось настроить HTTP-клиент provider."))?;
    let request = ChatCompletionRequest {
        model: config.model(),
        messages: [
            PromptMessage {
                role: "system",
                content: system_prompt,
            },
            PromptMessage {
                role: "user",
                content: user_prompt,
            },
        ],
        temperature: 0.0,
        max_completion_tokens: MAX_COMPLETION_TOKENS,
    };

    let response = client
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

    let completion = response
        .json::<ChatCompletionResponse>()
        .await
        .map_err(|_| anyhow::anyhow!("Provider вернул некорректный JSON."))?;
    let content = completion
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| anyhow::anyhow!("Provider не вернул текстовый ответ."))?;
    let content = content.trim();
    if content.is_empty() {
        bail!("Provider вернул пустой ответ.");
    }
    Ok(content.to_owned())
}
