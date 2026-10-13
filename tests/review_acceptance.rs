use anyhow::Result;
use developer_assistant::review::{
    corpus::load_project_corpus,
    embedding::{EMBEDDING_DIMENSIONS, Embedder, FastEmbedder},
    index::ReviewIndex,
};
use std::{
    collections::HashSet,
    fs,
    path::Path,
    process::{Command, Output},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tempfile::TempDir;

#[derive(Clone, Default)]
struct FixtureEmbedder {
    embedded_inputs: Arc<AtomicUsize>,
}

impl Embedder for FixtureEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedded_inputs
            .fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts
            .iter()
            .map(|text| {
                let lower = text.to_lowercase();
                let dimension = if lower.contains("transaction")
                    || lower.contains("balance")
                    || lower.contains("settlement")
                {
                    0
                } else {
                    1
                };
                let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
                embedding[dimension] = 1.0;
                embedding
            })
            .collect())
    }
}
#[derive(Default)]
struct OrthogonalEmbedder;

impl Embedder for OrthogonalEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let dimension = usize::from(text.to_ascii_lowercase().contains("orthogonal query"));
                let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
                embedding[dimension] = 1.0;
                embedding
            })
            .collect())
    }
}

#[test]
fn d32_review_filtered_rag() {
    let fixture = project_fixture();
    let root = fixture.path();
    let corpus = load_project_corpus(root).expect("filtered corpus loads");
    let paths = corpus
        .iter()
        .map(|document| document.path.as_str())
        .collect::<HashSet<_>>();

    assert!(paths.contains("README.md"));
    assert!(paths.contains("docs/architecture.md"));
    assert!(paths.contains("src/transaction.rs"));
    for excluded in [
        "ignored.txt",
        "src/config.rs",
        "credentials.txt",
        "tokens.json",
        "settings.json",
        "src/auth.rs",
        "build/generated.rs",
        "vendor/library.rs",
        "oversized.txt",
        "src/binary.dat",
        "src/non-utf8.txt",
        "src/alias.rs",
    ] {
        assert!(
            !paths.contains(excluded),
            "filtered corpus included {excluded}"
        );
    }

    let cache = root.join(".data/developer-assistant");
    let mut index = ReviewIndex::open(root, &cache, FixtureEmbedder::default())
        .expect("review index builds with a deterministic test embedder");
    let results = index
        .search("How does the importer validate transaction balances?", 8)
        .expect("hybrid retrieval succeeds");
    let result_paths = results
        .iter()
        .map(|result| result.path.as_str())
        .collect::<HashSet<_>>();
    assert!(result_paths.contains("docs/architecture.md"));
    assert!(result_paths.contains("src/transaction.rs"));
    assert!(!result_paths.contains("ignored.txt"));

    let excerpt = index
        .read_file("src/transaction.rs", 1, 2)
        .expect("allowlisted project file can be read");
    assert!(excerpt.contains("validate_transaction_balance"));
    assert!(excerpt.starts_with("    1 | "));
    assert!(index.read_file("../README.md", 1, 2).is_err());
    assert!(index.read_file("ignored.txt", 1, 2).is_err());
    assert!(index.read_file("src/transaction.rs", 1, 81).is_err());
}

#[test]
fn d32_review_lexical_identifier_retrieval() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".data/lexical-review");
    let mut index = ReviewIndex::open(root, &cache, OrthogonalEmbedder)
        .expect("review index builds with orthogonal test vectors");
    let results = index
        .search("orthogonal query validate_transaction_balance", 8)
        .expect("lexical retrieval handles an identifier absent from semantic vectors");

    assert_eq!(
        results.first().map(|result| result.path.as_str()),
        Some("src/transaction.rs")
    );
}

#[cfg(unix)]
#[test]
fn d32_review_cache_rejects_symlink_escape() {
    let fixture = project_fixture();
    let root = fixture.path();
    let outside = tempfile::tempdir().expect("outside directory is created");
    fs::create_dir(root.join(".data")).expect("local cache parent is created");
    std::os::unix::fs::symlink(outside.path(), root.join(".data/external"))
        .expect("cache escape symlink is created");

    let cache_dir = root.join(".data/external");
    let index_result = ReviewIndex::open(root, &cache_dir, OrthogonalEmbedder);
    let model_result = FastEmbedder::load(root, &cache_dir);
    assert!(index_result.is_err());
    assert!(model_result.is_err());
    assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
}

#[test]
fn d32_review_index_cache_matches_current_corpus() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".data/developer-assistant");
    let embedder = FixtureEmbedder::default();
    let count = Arc::clone(&embedder.embedded_inputs);

    let first = ReviewIndex::open(root, &cache, embedder.clone()).expect("initial index builds");
    let initial_embeddings = count.load(Ordering::Relaxed);
    assert!(initial_embeddings > 0);
    drop(first);

    let second = ReviewIndex::open(root, &cache, embedder.clone()).expect("matching cache loads");
    assert_eq!(count.load(Ordering::Relaxed), initial_embeddings);
    drop(second);

    write_text(
        root,
        "docs/architecture.md",
        "The transaction importer validates each account balance before producing its daily batch.\nThe revised pipeline checks duplicate transaction identifiers before persistence.\n",
    );
    let _updated =
        ReviewIndex::open(root, &cache, embedder).expect("changed corpus rebuilds index");
    assert!(count.load(Ordering::Relaxed) > initial_embeddings);
}

#[test]
fn d32_review_ignored_paths_do_not_deadlock() {
    let fixture = project_fixture();
    let root = fixture.path();
    write_text(root, ".gitignore", "ignored.txt\n.data/\nignored-many/\n");
    for index in 0..512 {
        let path = format!("ignored-many/{index:04}-{}.txt", "x".repeat(180));
        write_text(root, &path, "synthetic ignored content\n");
    }

    let add = git(root, &["add", "-f", "-A"]);
    assert!(
        add.status.success(),
        "git add failed: {}",
        String::from_utf8_lossy(&add.stderr)
    );
    let corpus = load_project_corpus(root).expect("many ignored paths are processed concurrently");
    assert!(corpus.iter().any(|document| document.path == "README.md"));
    assert!(
        corpus
            .iter()
            .all(|document| !document.path.starts_with("ignored-many/"))
    );
}

#[test]
fn d32_review_cache_requires_git_exclusion() {
    let fixture = project_fixture();
    let root = fixture.path();
    let cache = root.join(".review-cache");

    assert!(ReviewIndex::open(root, &cache, OrthogonalEmbedder).is_err());
    assert!(FastEmbedder::load(root, &cache).is_err());
    assert!(!cache.exists(), "unignored cache directory is not created");
}
fn project_fixture() -> TempDir {
    let fixture = tempfile::tempdir().expect("temporary project directory is created");
    let root = fixture.path();
    let init = git(root, &["init", "-q"]);
    assert!(
        init.status.success(),
        "git init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    write_text(root, ".gitignore", "ignored.txt\n.data/\n");
    write_text(
        root,
        "README.md",
        "Project overview and module boundaries.\n",
    );
    write_text(
        root,
        "docs/architecture.md",
        "The transaction importer validates each account balance before producing its daily batch.\n",
    );
    write_text(
        root,
        "src/transaction.rs",
        "pub fn validate_transaction_balance() {\n    validate_account_balance();\n}\n",
    );
    write_text(
        root,
        "docs/login.md",
        "A session cookie is created after a successful login.\n",
    );
    write_text(
        root,
        "ignored.txt",
        "transaction balance validation is documented here.\n",
    );
    write_text(
        root,
        "src/config.rs",
        "const API_KEY: &str = \"ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";\n",
    );
    write_text(
        root,
        "credentials.txt",
        "A credential file must never enter the project index.\n",
    );
    write_text(
        root,
        "tokens.json",
        "{\"token\":\"synthetic fixture value\"}\n",
    );
    write_text(
        root,
        "settings.json",
        r#"{"api_key":"0123456789abcdef0123456789abcdef"}"#,
    );
    write_text(
        root,
        "src/auth.rs",
        "Authorization: Bearer synthetic_token_value_0123456789abcdef\n",
    );
    write_text(root, "build/generated.rs", "generated transaction helper\n");
    write_text(root, "vendor/library.rs", "vendor transaction helper\n");
    write_text(root, "oversized.txt", &"x".repeat(256 * 1024 + 1));
    fs::write(root.join("src/binary.dat"), [0_u8, 1, 2, 3, 4]).expect("binary fixture is written");
    fs::write(root.join("src/non-utf8.txt"), [0xff, 0xfe, 0x80])
        .expect("non-UTF8 fixture is written");
    #[cfg(unix)]
    std::os::unix::fs::symlink("transaction.rs", root.join("src/alias.rs"))
        .expect("symlink fixture is created");

    let output = git(root, &["add", "-f", "-A"]);
    assert!(
        output.status.success(),
        "git add failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fixture
}

fn write_text(root: &Path, relative: &str, text: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("fixture path has a parent"))
        .expect("fixture parent is created");
    fs::write(path, text).expect("fixture file is written");
}

fn git(root: &Path, arguments: &[&str]) -> Output {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .expect("git process starts")
}
