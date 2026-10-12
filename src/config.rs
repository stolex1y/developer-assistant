use anyhow::{Result, bail};
use reqwest::Url;
use std::{env, net::IpAddr, str::FromStr};

const API_KEY_ENV: &str = "DEVELOPER_ASSISTANT_API_KEY";
const BASE_URL_ENV: &str = "DEVELOPER_ASSISTANT_BASE_URL";
const MODEL_ENV: &str = "DEVELOPER_ASSISTANT_MODEL";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o-mini";

pub struct ProviderConfig {
    endpoint: Url,
    api_key: String,
    model: String,
}

impl ProviderConfig {
    pub fn from_env() -> Result<Self> {
        let api_key = env::var(API_KEY_ENV).ok();
        let base_url = env::var(BASE_URL_ENV).ok();
        let model = env::var(MODEL_ENV).ok();
        Self::from_values(api_key, base_url, model)
    }

    fn from_values(
        api_key: Option<String>,
        base_url: Option<String>,
        model: Option<String>,
    ) -> Result<Self> {
        let api_key = api_key
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.trim().to_owned())
            .ok_or_else(|| anyhow::anyhow!("Задайте API key в переменной {API_KEY_ENV}."))?;

        let base_url = base_url
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let mut endpoint = Url::parse(base_url.trim())
            .map_err(|_| anyhow::anyhow!("Некорректный provider base URL."))?;

        if !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            bail!("Provider base URL не должен содержать credentials, query или fragment.");
        }

        let host = endpoint
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("Provider base URL должен содержать host."))?;
        if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && is_loopback_host(host))
        {
            bail!("Удалённый provider должен использовать HTTPS.");
        }

        let path = format!("{}/chat/completions", endpoint.path().trim_end_matches('/'));
        endpoint.set_path(&path);

        let model = model
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned());

        Ok(Self {
            endpoint,
            api_key,
            model,
        })
    }

    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || IpAddr::from_str(host.trim_matches(['[', ']']))
            .is_ok_and(|address| address.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::ProviderConfig;

    #[test]
    fn defaults_to_openai_compatible_chat_completions() {
        let config = ProviderConfig::from_values(Some("test-key".into()), None, None).unwrap();
        assert_eq!(
            config.endpoint().as_str(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(config.model(), "gpt-4o-mini");
    }

    #[test]
    fn allows_http_only_for_loopback_test_providers() {
        let config = ProviderConfig::from_values(
            Some("test-key".into()),
            Some("http://127.0.0.1:8080/v1".into()),
            Some("fixture".into()),
        )
        .unwrap();
        assert_eq!(
            config.endpoint().as_str(),
            "http://127.0.0.1:8080/v1/chat/completions"
        );

        assert!(
            ProviderConfig::from_values(
                Some("test-key".into()),
                Some("http://provider.example/v1".into()),
                Some("fixture".into()),
            )
            .is_err()
        );
    }

    #[test]
    fn refuses_credentials_embedded_in_base_url() {
        assert!(
            ProviderConfig::from_values(
                Some("test-key".into()),
                Some("https://user:secret@example.com/v1".into()),
                None,
            )
            .is_err()
        );
    }
    #[test]
    fn requires_api_key_without_disclosing_a_value() {
        let error = ProviderConfig::from_values(None, None, None)
            .err()
            .expect("missing API key is rejected");
        assert!(error.to_string().contains("DEVELOPER_ASSISTANT_API_KEY"));
    }
}
