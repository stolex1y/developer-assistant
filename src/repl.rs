use crate::{
    assistant::{answer_question, escape_terminal_field, print_result},
    config::{UserConfig, example_config_path},
};
use anyhow::Result;
use std::{
    io::{self, Write},
    path::Path,
};

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

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        let model_label = active_model
            .as_deref()
            .map(escape_terminal_field)
            .unwrap_or_else(|| "модель не выбрана".into());
        print!("developer-assistant [{model_label}]> ");
        io::stdout().flush()?;

        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let command = line.trim();
        if command.is_empty() {
            continue;
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
            match answer_question(project_root, question, config, active_model.as_deref()).await {
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
        _ => Err(anyhow::anyhow!(
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
