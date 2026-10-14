use crate::{
    assistant::{answer_question, escape_terminal_field, print_result},
    config::{UserConfig, example_config_path},
    retrieval::ProjectIndex,
};
use anyhow::{Result, anyhow};
use rustyline::{
    Config, Editor,
    completion::{Completer, Pair},
    error::ReadlineError,
    highlight::Highlighter,
    hint::Hinter,
    validate::Validator,
};
use std::path::Path;

const COMMANDS: [&str; 4] = ["/help", "/models", "/model", "/exit"];

pub async fn run(
    project_root: &Path,
    config: &mut UserConfig,
    mut active_model: Option<String>,
) -> Result<()> {
    println!("Ассистент разработчика. Команды: /models, /model, /help <вопрос>, /exit.");
    if config.models().next().is_none() {
        println!(
            "Каталог моделей пуст. Config: {}. Пример: {}.",
            escape_terminal_field(&config.config_path().display().to_string()),
            escape_terminal_field(&example_config_path().display().to_string())
        );
    }

    let readline_config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor = Editor::<SlashHelper, rustyline::history::MemHistory>::with_history(
        readline_config,
        rustyline::history::MemHistory::new(),
    )?;
    editor.set_helper(Some(SlashHelper));
    let mut project_index = ProjectIndex::default();

    loop {
        let model_label = active_model
            .as_deref()
            .map(escape_terminal_field)
            .unwrap_or_else(|| "модель не выбрана".into());
        let prompt = format!("developer-assistant [{model_label}]> ");
        let line = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(error) => return Err(anyhow!("Не удалось прочитать команду: {error}")),
        };
        let command = line.trim();
        if command.is_empty() {
            continue;
        }
        if let Some(entry) = history_entry(command) {
            editor.add_history_entry(entry.to_owned())?;
        }
        if command == "/exit" {
            break;
        }
        if command == "/models" {
            print_models(config, active_model.as_deref());
            continue;
        }
        if let Some(question) = help_question(command) {
            if question.is_empty() {
                println!("Использование: /help <вопрос>");
                continue;
            }
            match answer_question(
                project_root,
                question,
                config,
                active_model.as_deref(),
                &mut project_index,
            )
            .await
            {
                Ok(result) => print_result(result),
                Err(error) => eprintln!("Ошибка: {}", escape_terminal_field(&error.to_string())),
            }
            continue;
        }
        if command == "/model" || command.starts_with("/model ") || command.starts_with("/model\t")
        {
            match apply_model_command(command, config) {
                Ok(ModelCommandResult::Session(model)) => {
                    println!("Модель текущей сессии: {}", escape_terminal_field(&model));
                    active_model = Some(model);
                }
                Ok(ModelCommandResult::Default(model)) => {
                    println!(
                        "Default model сохранена и выбрана: {}",
                        escape_terminal_field(&model)
                    );
                    active_model = Some(model);
                }
                Err(error) => {
                    eprintln!("Ошибка: {}", escape_terminal_field(&error.to_string()));
                }
            }
            continue;
        }
        println!("Неизвестная команда. Используйте /models, /model, /help или /exit.");
    }
    Ok(())
}

struct SlashHelper;

impl Completer for SlashHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        position: usize,
        _context: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let prefix = &line[..position];
        if !prefix.starts_with('/') || prefix.chars().any(char::is_whitespace) {
            return Ok((position, Vec::new()));
        }
        let candidates = COMMANDS
            .iter()
            .filter(|command| command.starts_with(prefix))
            .map(|command| Pair {
                display: (*command).to_owned(),
                replacement: (*command).to_owned(),
            })
            .collect();
        Ok((0, candidates))
    }
}

impl Hinter for SlashHelper {
    type Hint = String;
}

impl Highlighter for SlashHelper {}
impl Validator for SlashHelper {}
impl rustyline::Helper for SlashHelper {}

enum ModelCommandResult {
    Session(String),
    Default(String),
}

fn apply_model_command(command: &str, config: &mut UserConfig) -> Result<ModelCommandResult> {
    let mut parts = command.split_whitespace();
    let _command = parts.next();
    let first = parts.next();
    let second = parts.next();
    let extra = parts.next();

    match (first, second, extra) {
        (Some("default"), Some(model_id), None) => {
            let model = config.resolve_model(model_id)?.qualified_id();
            config.set_default_model(&model)?;
            Ok(ModelCommandResult::Default(model))
        }
        (Some(model_id), None, None) => {
            let model = config.resolve_model(model_id)?.qualified_id();
            Ok(ModelCommandResult::Session(model))
        }
        _ => Err(anyhow!(
            "Использование: /model <provider/model> или /model default <provider/model>."
        )),
    }
}

fn print_models(config: &UserConfig, active_model: Option<&str>) {
    if config.models().next().is_none() {
        println!(
            "Каталог моделей пуст. Config: {}. Пример: {}.",
            escape_terminal_field(&config.config_path().display().to_string()),
            escape_terminal_field(&example_config_path().display().to_string())
        );
        return;
    }

    println!("Настроенные модели:");
    for (provider_id, model_id) in config.models() {
        let is_default = model_matches(config.default_model(), provider_id, model_id);
        let is_active = model_matches(active_model, provider_id, model_id);
        let suffix = match (is_default, is_active) {
            (true, true) => " [default, сессия]",
            (true, false) => " [default]",
            (false, true) => " [сессия]",
            (false, false) => "",
        };
        println!(
            "- {}/{}{}",
            escape_terminal_field(provider_id),
            escape_terminal_field(model_id),
            suffix
        );
    }
}

fn model_matches(selection: Option<&str>, provider_id: &str, model_id: &str) -> bool {
    selection
        .and_then(|selection| selection.split_once('/'))
        .is_some_and(|(selected_provider, selected_model)| {
            selected_provider == provider_id && selected_model == model_id
        })
}

fn help_question(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("/help")?;
    if !rest.is_empty() && !rest.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    Some(rest.trim())
}
fn history_entry(line: &str) -> Option<&str> {
    help_question(line)
        .filter(|question| !question.is_empty())
        .map(|_| line)
}

#[cfg(test)]
mod tests {
    use super::{SlashHelper, history_entry};
    use rustyline::{Context, completion::Completer, history::MemHistory};

    #[test]
    fn tab_completes_only_slash_commands_before_arguments() {
        let history = MemHistory::new();
        let context = Context::new(&history);
        let (start, single) = SlashHelper.complete("/he", 3, &context).unwrap();
        assert_eq!(start, 0);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].replacement, "/help");

        let (_, ambiguous) = SlashHelper.complete("/m", 2, &context).unwrap();
        assert_eq!(
            ambiguous
                .iter()
                .map(|candidate| candidate.replacement.as_str())
                .collect::<Vec<_>>(),
            ["/models", "/model"]
        );
        let (_, exit_command) = SlashHelper.complete("/ex", 3, &context).unwrap();
        assert_eq!(exit_command.len(), 1);
        assert_eq!(exit_command[0].replacement, "/exit");
        assert!(
            SlashHelper
                .complete("/help ", 6, &context)
                .unwrap()
                .1
                .is_empty()
        );
        assert!(
            SlashHelper
                .complete("project", 7, &context)
                .unwrap()
                .1
                .is_empty()
        );
    }

    #[test]
    fn only_nonempty_help_questions_enter_session_history() {
        assert_eq!(
            history_entry("/help как устроен проект?"),
            Some("/help как устроен проект?")
        );
        assert_eq!(history_entry("/help"), None);
        assert_eq!(history_entry("/models"), None);
        assert_eq!(history_entry("/model deepseek/model"), None);
        assert_eq!(history_entry("/exit"), None);
        assert_eq!(history_entry("/helpful text"), None);
    }
}
