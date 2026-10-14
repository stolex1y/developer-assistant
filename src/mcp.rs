use crate::git_process::{isolate_std_git_environment, isolate_tokio_git_environment};
use anyhow::{Context, Result, anyhow, bail};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, CallToolResult, ErrorData},
    tool, tool_router,
    transport::{TokioChildProcess, io::stdio},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output as StdOutput, Stdio},
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

const MAX_LISTED_FILES: usize = 50;

#[derive(Debug, Deserialize, Serialize)]
pub struct FileInventory {
    pub total: usize,
    pub paths: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug)]
pub struct ProjectContext {
    pub branch: String,
    pub files: Option<FileInventory>,
    pub file_list_warning: Option<&'static str>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BranchResult {
    branch: String,
}

pub fn canonical_git_root(project_root: &Path) -> Result<PathBuf> {
    let root = std::fs::canonicalize(project_root).with_context(|| {
        format!(
            "Не удалось открыть project root {}.",
            project_root.display()
        )
    })?;
    let output = run_git_sync(&root, &["rev-parse", "--show-toplevel"])
        .context("Не удалось проверить Git root проекта.")?;
    if !output.status.success() {
        bail!("Указанная директория не является Git working tree.");
    }
    let top_level = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let top_level =
        std::fs::canonicalize(top_level).context("Git не вернул доступный корень проекта.")?;
    if top_level != root {
        bail!("Укажите корень Git repository, а не вложенную директорию.");
    }
    Ok(root)
}

pub async fn serve_stdio(project_root: &Path) -> Result<()> {
    let root = canonical_git_root(project_root)?;
    ProjectTools { root }
        .serve(stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}

pub async fn read_project_context<'a>(
    project_root: &Path,
    provider_key_env_names: impl IntoIterator<Item = &'a str>,
) -> Result<ProjectContext> {
    let executable =
        std::env::current_exe().context("Не удалось определить путь CLI executable.")?;
    let mut command = Command::new(executable);
    command
        .arg("__mcp-server")
        .arg("--project-root")
        .arg(project_root)
        .env_remove("DEVELOPER_ASSISTANT_API_KEY")
        .env_remove("DEVELOPER_ASSISTANT_BASE_URL")
        .env_remove("DEVELOPER_ASSISTANT_MODEL");
    for env_name in provider_key_env_names {
        command.env_remove(env_name);
    }
    let client = ()
        .serve(
            TokioChildProcess::new(command)
                .context("Не удалось запустить локальный MCP server.")?,
        )
        .await
        .map_err(|_| anyhow!("Не удалось подключиться к локальному MCP server."))?;

    let result = async {
        let branch = client
            .call_tool(CallToolRequestParams::new("get_current_branch"))
            .await
            .map_err(|_| anyhow!("MCP не вернул текущую Git-ветку."))?
            .into_typed::<BranchResult>()
            .map_err(|_| anyhow!("MCP вернул некорректный ответ о Git-ветке."))?;
        if branch.branch.trim().is_empty() {
            bail!("Git repository находится в состоянии detached HEAD; текущей ветки нет.");
        }

        let file_result = client
            .call_tool(CallToolRequestParams::new("list_files"))
            .await;
        let files = file_result
            .ok()
            .and_then(|result| result.into_typed::<FileInventory>().ok());
        let file_list_warning = files
            .is_none()
            .then_some("MCP не вернул список файлов; ответ продолжен только с Git-веткой.");

        Ok(ProjectContext {
            branch: branch.branch,
            files,
            file_list_warning,
        })
    }
    .await;

    let shutdown = client.cancel().await;
    match result {
        Err(error) => Err(error),
        Ok(context) => {
            shutdown.map_err(|_| anyhow!("Не удалось штатно закрыть локальный MCP server."))?;
            Ok(context)
        }
    }
}

#[derive(Debug, Clone)]
struct ProjectTools {
    root: PathBuf,
}

#[tool_router(server_handler)]
impl ProjectTools {
    #[tool(
        name = "get_current_branch",
        description = "Return the current Git branch of the selected project repository."
    )]
    async fn get_current_branch(&self) -> std::result::Result<CallToolResult, ErrorData> {
        let output = run_git_async(&self.root, &["branch", "--show-current"])
            .await
            .map_err(|_| ErrorData::internal_error("Не удалось прочитать Git-ветку.", None))?;
        if !output.status.success() {
            return Err(ErrorData::internal_error(
                "Не удалось прочитать Git-ветку.",
                None,
            ));
        }
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if branch.is_empty() {
            return Err(ErrorData::internal_error(
                "В repository нет текущей Git-ветки (detached HEAD).",
                None,
            ));
        }
        Ok(CallToolResult::structured(json!({ "branch": branch })))
    }

    #[tool(
        name = "list_files",
        description = "List at most 50 tracked Git-index paths without reading file contents."
    )]
    async fn list_files(&self) -> std::result::Result<CallToolResult, ErrorData> {
        let inventory = read_git_file_inventory(&self.root).await.map_err(|_| {
            ErrorData::internal_error("Не удалось получить список Git-файлов.", None)
        })?;
        Ok(CallToolResult::structured(json!(inventory)))
    }
}

async fn read_git_file_inventory(root: &Path) -> std::io::Result<FileInventory> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--cached", "--deduplicate", "-z"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    isolate_tokio_git_environment(&mut command);
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("Git file-list output is unavailable."))?;
    let mut reader = BufReader::new(stdout);
    let mut path_bytes = Vec::new();
    let mut paths = Vec::with_capacity(MAX_LISTED_FILES);
    let mut total = 0;

    loop {
        path_bytes.clear();
        if reader.read_until(0, &mut path_bytes).await? == 0 {
            break;
        }
        if path_bytes.last() == Some(&0) {
            path_bytes.pop();
        }
        if path_bytes.is_empty() {
            continue;
        }
        total += 1;
        if paths.len() < MAX_LISTED_FILES {
            paths.push(String::from_utf8_lossy(&path_bytes).into_owned());
        }
    }

    if !child.wait().await?.success() {
        return Err(std::io::Error::other("Git file-list command failed."));
    }
    let truncated = total > paths.len();
    Ok(FileInventory {
        total,
        paths,
        truncated,
    })
}

fn run_git_sync(root: &Path, args: &[&str]) -> std::io::Result<StdOutput> {
    let mut command = StdCommand::new("git");
    command.arg("-C").arg(root).args(args);
    isolate_std_git_environment(&mut command);
    command.output()
}

async fn run_git_async(root: &Path, args: &[&str]) -> std::io::Result<std::process::Output> {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(args);
    isolate_tokio_git_environment(&mut command);
    command.output().await
}
