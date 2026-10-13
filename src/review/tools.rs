use crate::review::{
    corpus::contains_secret_like_value,
    deepseek::ReviewToolExecutor,
    embedding::Embedder,
    index::{RetrievedPassage, ReviewIndex},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct IndexTools<'a, E> {
    index: &'a mut ReviewIndex<E>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchCodeInput {
    query: String,
    limit: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileInput {
    path: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Serialize)]
struct SearchCodeOutput<'a> {
    results: Vec<SearchCodeResult<'a>>,
}

#[derive(Serialize)]
struct SearchCodeResult<'a> {
    path: &'a str,
    start_line: usize,
    end_line: usize,
    text: &'a str,
}

impl<'a, E: Embedder> IndexTools<'a, E> {
    pub fn new(index: &'a mut ReviewIndex<E>) -> Self {
        Self { index }
    }

    fn search_code(&mut self, arguments: Value) -> Result<String> {
        let input: SearchCodeInput =
            serde_json::from_value(arguments).context("Некорректные аргументы search_code.")?;
        if input.query.trim().is_empty()
            || input.query.len() > 1000
            || input.query.chars().any(char::is_control)
            || !(1..=8).contains(&input.limit)
        {
            bail!("search_code принимает непустой query до 1000 байт и limit от 1 до 8.");
        }

        let results = self.index.search(&input.query, input.limit)?;
        if results
            .iter()
            .any(|passage| contains_secret_like_value(&passage.text))
        {
            bail!("search_code обнаружил secret-like passage; результат не передан модели.");
        }
        let output = SearchCodeOutput {
            results: results.iter().map(SearchCodeResult::from).collect(),
        };
        serde_json::to_string(&output).context("Не удалось сформировать search_code result.")
    }

    fn read_file(&self, arguments: Value) -> Result<String> {
        let input: ReadFileInput =
            serde_json::from_value(arguments).context("Некорректные аргументы read_file.")?;
        if input.path.is_empty()
            || input.path.len() > 1024
            || input.path.chars().any(char::is_control)
            || input.start_line == 0
            || input.end_line < input.start_line
            || input.end_line - input.start_line >= 80
        {
            bail!("read_file принимает безопасный путь и диапазон не более 80 строк.");
        }
        let content = self
            .index
            .read_file(&input.path, input.start_line, input.end_line)?;
        if contains_secret_like_value(&content) {
            bail!("read_file обнаружил secret-like text; результат не передан модели.");
        }
        Ok(content)
    }
}

impl<E: Embedder> ReviewToolExecutor for IndexTools<'_, E> {
    fn execute(&mut self, name: &str, arguments: Value) -> Result<String> {
        match name {
            "search_code" => self.search_code(arguments),
            "read_file" => self.read_file(arguments),
            _ => bail!("Неизвестный read-only review tool."),
        }
    }
}

impl<'a> From<&'a RetrievedPassage> for SearchCodeResult<'a> {
    fn from(passage: &'a RetrievedPassage) -> Self {
        Self {
            path: &passage.path,
            start_line: passage.start_line,
            end_line: passage.end_line,
            text: &passage.text,
        }
    }
}
