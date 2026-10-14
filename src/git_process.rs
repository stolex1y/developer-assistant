use std::process::Command as StdCommand;
use tokio::process::Command as TokioCommand;

const GIT_ENV_TO_CLEAR: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
];

pub(crate) fn isolate_std_git_environment(command: &mut StdCommand) {
    for variable in GIT_ENV_TO_CLEAR {
        command.env_remove(variable);
    }
    command.env("GIT_OPTIONAL_LOCKS", "0");
}

pub(crate) fn isolate_tokio_git_environment(command: &mut TokioCommand) {
    for variable in GIT_ENV_TO_CLEAR {
        command.env_remove(variable);
    }
    command.env("GIT_OPTIONAL_LOCKS", "0");
}
