use anyhow::{Result, anyhow, bail};
use reqwest::Url;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    env,
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    net::IpAddr,
    path::{Path, PathBuf},
    process,
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
};
use toml_edit::{DocumentMut, Item, Table, value};

#[derive(Debug)]
pub struct UserConfig {
    path: PathBuf,
    data: ConfigFile,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigFile {
    providers: BTreeMap<String, ProviderDefinition>,
    selection: SelectionConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderDefinition {
    base_url: String,
    api_key_env: String,
    models: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SelectionConfig {
    default_model: Option<String>,
}

#[derive(Debug)]
pub struct ModelSelection<'a> {
    provider_id: &'a str,
    model_id: &'a str,
    provider: &'a ProviderDefinition,
}

pub struct ProviderConfig {
    endpoint: Url,
    api_key: String,
    model: String,
}

impl UserConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = match path {
            Some(path) => path.to_owned(),
            None => default_config_path()?,
        };
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(Self {
                    path,
                    data: ConfigFile::default(),
                });
            }
            Err(_) => bail!("Не удалось прочитать пользовательский config."),
        };
        let data = toml_edit::de::from_str::<ConfigFile>(&contents)
            .map_err(|_| anyhow!("Некорректная структура пользовательского config."))?;
        data.validate()?;

        Ok(Self { path, data })
    }

    pub fn config_path(&self) -> &Path {
        &self.path
    }

    pub fn models(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.data
            .providers
            .iter()
            .flat_map(|(provider_id, provider)| {
                provider
                    .models
                    .iter()
                    .map(move |model_id| (provider_id.as_str(), model_id.as_str()))
            })
    }

    pub fn default_model(&self) -> Option<&str> {
        self.data.selection.default_model.as_deref()
    }

    pub fn provider_key_env_names(&self) -> impl Iterator<Item = &str> {
        self.data
            .providers
            .values()
            .map(|provider| provider.api_key_env.as_str())
    }

    pub fn resolve_model(&self, qualified_id: &str) -> Result<ModelSelection<'_>> {
        self.data.resolve_model(qualified_id)
    }

    pub fn set_default_model(&mut self, qualified_id: &str) -> Result<()> {
        let selected_id = self.resolve_model(qualified_id)?.qualified_id();
        if !self.path.exists() {
            bail!("Config-файл отсутствует; сохранение не создаёт его автоматически.");
        }

        let contents = fs::read_to_string(&self.path)
            .map_err(|_| anyhow!("Не удалось прочитать config для сохранения default."))?;
        let mut document = contents
            .parse::<DocumentMut>()
            .map_err(|_| anyhow!("Некорректный TOML в пользовательском config."))?;
        let mut updated = toml_edit::de::from_str::<ConfigFile>(&contents)
            .map_err(|_| anyhow!("Некорректная структура пользовательского config."))?;
        updated.validate()?;
        updated.resolve_model(&selected_id)?;

        let selection_item = document
            .as_table_mut()
            .entry("selection")
            .or_insert(Item::Table(Table::new()));
        match selection_item {
            Item::Table(selection_table) => {
                selection_table.insert("default_model", value(selected_id.clone()));
            }
            Item::Value(selection_value) => {
                let selection_table = selection_value
                    .as_inline_table_mut()
                    .ok_or_else(|| anyhow!("Поле selection в config должно быть таблицей."))?;
                selection_table
                    .insert("default_model", toml_edit::Value::from(selected_id.clone()));
            }
            _ => bail!("Поле selection в config должно быть таблицей."),
        }

        updated.selection.default_model = Some(selected_id);
        updated.validate()?;
        atomic_write(&self.path, &document.to_string())?;
        self.data = updated;
        Ok(())
    }
}

impl ConfigFile {
    fn validate(&self) -> Result<()> {
        for (provider_id, provider) in &self.providers {
            validate_provider_id(provider_id)?;
            validate_env_name(&provider.api_key_env)?;
            endpoint_for_base_url(&provider.base_url)?;
            if provider.models.is_empty() {
                bail!("У каждого provider должна быть задана хотя бы одна модель.");
            }

            let mut models = HashSet::new();
            for model_id in &provider.models {
                validate_model_id(model_id)?;
                if !models.insert(model_id.as_str()) {
                    bail!("В каталоге provider повторяется model ID.");
                }
            }
        }

        if let Some(default_model) = self.selection.default_model.as_deref() {
            self.resolve_model(default_model)
                .map_err(|_| anyhow!("Default model не входит в каталог providers/models."))?;
        }
        Ok(())
    }

    fn resolve_model(&self, qualified_id: &str) -> Result<ModelSelection<'_>> {
        let (requested_provider_id, requested_model_id) = qualified_id
            .split_once('/')
            .ok_or_else(|| anyhow!("Ожидается ID модели в формате provider/model."))?;
        let (provider_id, provider) = self
            .providers
            .get_key_value(requested_provider_id)
            .ok_or_else(|| anyhow!("Неизвестная пара provider/model; используйте /models."))?;
        let model_id = provider
            .models
            .iter()
            .find(|candidate| candidate.as_str() == requested_model_id)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("Неизвестная пара provider/model; используйте /models."))?;
        Ok(ModelSelection {
            provider_id,
            model_id,
            provider,
        })
    }
}

impl ModelSelection<'_> {
    pub fn provider_id(&self) -> &str {
        self.provider_id
    }

    pub fn model_id(&self) -> &str {
        self.model_id
    }

    pub fn qualified_id(&self) -> String {
        format!("{}/{}", self.provider_id, self.model_id)
    }

    pub fn provider_config(&self) -> Result<ProviderConfig> {
        ProviderConfig::from_selection(self)
    }
}

impl ProviderConfig {
    fn from_selection(selection: &ModelSelection<'_>) -> Result<Self> {
        let api_key = env::var(&selection.provider.api_key_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.trim().to_owned())
            .ok_or_else(|| {
                anyhow!(
                    "Задайте API key в переменной {}.",
                    selection.provider.api_key_env
                )
            })?;

        Ok(Self {
            endpoint: endpoint_for_base_url(&selection.provider.base_url)?,
            api_key,
            model: selection.model_id.to_owned(),
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

    #[cfg(test)]
    pub(crate) fn for_test(endpoint: &str) -> Self {
        Self {
            endpoint: Url::parse(endpoint).expect("test endpoint is valid"),
            api_key: "test-key".to_owned(),
            model: "test-model".to_owned(),
        }
    }
}

pub fn default_config_path() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .ok_or_else(|| anyhow!("Не удалось определить домашний каталог пользователя."))?;
    Ok(PathBuf::from(home)
        .join(".config")
        .join("developer-assistant")
        .join("config.toml"))
}

pub fn example_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.example.toml")
}

fn validate_provider_id(provider_id: &str) -> Result<()> {
    if provider_id.is_empty()
        || !provider_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_.-".contains(character))
    {
        bail!("Provider ID должен содержать только ASCII-буквы, цифры, '.', '_' или '-'.");
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> Result<()> {
    if model_id.is_empty()
        || model_id.trim() != model_id
        || model_id
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
    {
        bail!("Model ID не должен быть пустым или содержать пробелы/управляющие символы.");
    }
    Ok(())
}

fn validate_env_name(name: &str) -> Result<()> {
    let mut characters = name.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
    if !valid_first
        || !characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        bail!("api_key_env должен быть допустимым именем переменной окружения.");
    }
    Ok(())
}

fn endpoint_for_base_url(base_url: &str) -> Result<Url> {
    if base_url.trim().is_empty() {
        bail!("Provider base_url не должен быть пустым.");
    }
    let mut endpoint =
        Url::parse(base_url.trim()).map_err(|_| anyhow!("Некорректный provider base_url."))?;

    if !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        bail!("Provider base_url не должен содержать credentials, query или fragment.");
    }

    let host = endpoint
        .host_str()
        .ok_or_else(|| anyhow!("Provider base_url должен содержать host."))?;
    if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && is_loopback_host(host)) {
        bail!("Удалённый provider должен использовать HTTPS.");
    }

    let path = format!("{}/chat/completions", endpoint.path().trim_end_matches('/'));
    endpoint.set_path(&path);
    Ok(endpoint)
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || IpAddr::from_str(host.trim_matches(['[', ']']))
            .is_ok_and(|address| address.is_loopback())
}

fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("Не удалось определить имя пользовательского config."))?
        .to_string_lossy();

    for _ in 0..8 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_path =
            parent.join(format!(".{file_name}.{}.{}.tmp", process::id(), sequence));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(_) => bail!("Не удалось подготовить безопасную запись config."),
        };

        let write_succeeded = file
            .write_all(contents.as_bytes())
            .and_then(|()| file.sync_all())
            .is_ok();
        drop(file);
        if !write_succeeded || fs::rename(&temporary_path, path).is_err() {
            let _ = fs::remove_file(&temporary_path);
            bail!("Не удалось безопасно сохранить пользовательский config.");
        }
        return Ok(());
    }

    bail!("Не удалось подготовить временный файл для config.");
}

#[cfg(test)]
mod tests {
    use super::{UserConfig, default_config_path};
    use std::path::PathBuf;

    const DEEPSEEK_CONFIG: &str = include_str!("../config.example.toml");

    fn load_contents(contents: &str) -> anyhow::Result<UserConfig> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("config.toml");
        std::fs::write(&path, contents)?;
        // Keep the directory alive for the duration of parsing only.
        UserConfig::load(Some(&path))
    }

    #[test]
    fn loads_deepseek_catalog_and_default() {
        let config = load_contents(DEEPSEEK_CONFIG).expect("valid DeepSeek catalog");

        assert_eq!(
            config.models().collect::<Vec<_>>(),
            vec![
                ("deepseek", "deepseek-flash"),
                ("deepseek", "deepseek-v4-pro")
            ]
        );
        assert_eq!(config.default_model(), Some("deepseek/deepseek-flash"));
    }

    #[test]
    fn resolves_configured_provider_endpoint_and_model() {
        let config = load_contents(DEEPSEEK_CONFIG).expect("valid DeepSeek catalog");
        let selection = config
            .resolve_model("deepseek/deepseek-v4-pro")
            .expect("configured model resolves");

        assert_eq!(selection.provider_id(), "deepseek");
        assert_eq!(selection.model_id(), "deepseek-v4-pro");
        assert_eq!(selection.qualified_id(), "deepseek/deepseek-v4-pro");
    }

    #[test]
    fn persists_default_in_an_inline_selection_table() {
        let directory = tempfile::tempdir().expect("temp directory");
        let path = directory.path().join("config.toml");
        let contents = "selection = { default_model = \"deepseek/deepseek-flash\" }\n\n[providers.deepseek]\nbase_url = \"https://api.deepseek.com\"\napi_key_env = \"DEEPSEEK_API_KEY\"\nmodels = [\"deepseek-flash\", \"deepseek-v4-pro\"]\n";
        std::fs::write(&path, contents).expect("inline-table config is written");
        let mut config = UserConfig::load(Some(&path)).expect("inline-table config loads");

        config
            .set_default_model("deepseek/deepseek-v4-pro")
            .expect("inline-table default updates");

        assert_eq!(config.default_model(), Some("deepseek/deepseek-v4-pro"));
        let saved = std::fs::read_to_string(&path).expect("updated config is readable");
        assert!(saved.contains("deepseek-v4-pro"));
        assert!(saved.contains("api_key_env = \"DEEPSEEK_API_KEY\""));
        let reloaded = UserConfig::load(Some(&path)).expect("updated config reloads");
        assert_eq!(reloaded.default_model(), Some("deepseek/deepseek-v4-pro"));
    }

    #[test]
    fn rejects_default_not_in_catalog() {
        let invalid = DEEPSEEK_CONFIG.replace("deepseek/deepseek-flash", "deepseek/unknown");
        let error = load_contents(&invalid).expect_err("unknown default is rejected");

        assert!(error.to_string().contains("Default model"));
    }

    #[test]
    fn rejects_remote_http_provider() {
        let invalid = DEEPSEEK_CONFIG.replace("https://api.deepseek.com", "http://example.com");
        let error = load_contents(&invalid).expect_err("remote cleartext endpoint is rejected");

        assert!(error.to_string().contains("HTTPS"));
    }

    #[test]
    fn rejects_unlisted_model_without_echoing_input() {
        let config = load_contents(DEEPSEEK_CONFIG).expect("valid DeepSeek catalog");
        let error = config
            .resolve_model("deepseek/unlisted\u{001b}[2J")
            .expect_err("unlisted model is rejected");

        assert!(error.to_string().contains("Неизвестная пара"));
        assert!(!error.to_string().contains('\u{001b}'));
    }

    #[test]
    fn uses_requested_config_path_without_creating_missing_file() {
        let directory = tempfile::tempdir().expect("temp directory");
        let path = directory.path().join("missing.toml");
        let config = UserConfig::load(Some(&path)).expect("missing config is empty");

        assert_eq!(config.config_path(), PathBuf::from(&path));
        assert_eq!(config.models().count(), 0);
        assert!(!path.exists());
    }

    #[test]
    fn defaults_to_user_config_path_under_home() {
        let home = std::env::var_os("HOME").expect("test environment has HOME");
        assert_eq!(
            default_config_path().expect("default config path"),
            PathBuf::from(home)
                .join(".config")
                .join("developer-assistant")
                .join("config.toml")
        );
    }
}
