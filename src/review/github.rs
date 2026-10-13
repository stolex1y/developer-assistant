use crate::review::report::{has_summary_marker, parse_finding_fingerprint};
use anyhow::{Context, Result, bail};
use reqwest::{Client, Response, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{net::IpAddr, str::FromStr, time::Duration};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_CHANGED_FILES: usize = 3_000;
const MAX_DIFF_BYTES: usize = 8 * 1024 * 1024;
const PAGE_SIZE: u64 = 100;
const MAX_ISSUE_COMMENT_PAGES: u64 = 10;
const MAX_REVIEW_THREADS: usize = 1_000;
const MAX_FINDING_COMMENT_BYTES: usize = 16 * 1024;
const VIEWER_QUERY: &str = "query { viewer { login } }";

const THREADS_QUERY: &str = r#"
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 1) {
            nodes {
              databaseId
              body
              author { login }
            }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"#;

const RESOLVE_THREAD_MUTATION: &str = r#"
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { id isResolved }
  }
}
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullRequestId {
    pub owner: String,
    pub repository: String,
    pub number: u64,
    pub expected_base_sha: String,
    pub expected_head_sha: String,
}

impl PullRequestId {
    pub fn validate(&self) -> Result<()> {
        if !valid_repository_component(&self.owner)
            || !valid_repository_component(&self.repository)
            || self.number == 0
            || self.number > i32::MAX as u64
            || !valid_sha(&self.expected_base_sha)
            || !valid_sha(&self.expected_head_sha)
        {
            bail!("Pull request identity или ожидаемые commit SHA некорректны.");
        }
        Ok(())
    }

    fn full_name(&self) -> String {
        format!("{}/{}", self.owner, self.repository)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChangedFile {
    pub path: String,
    pub status: String,
    pub patch: Option<String>,
    pub added_lines: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullRequestSnapshot {
    pub base_sha: String,
    pub head_sha: String,
    pub files: Vec<ChangedFile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExistingThread {
    pub thread_id: String,
    pub comment_id: u64,
    pub fingerprint: String,
    pub path: String,
    pub line: Option<usize>,
    pub body: String,
    pub is_resolved: bool,
    pub is_outdated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExistingSummary {
    pub comment_id: u64,
}

pub struct GitHubClient {
    api_base: Url,
    token: String,
    http: Client,
}

#[derive(Deserialize)]
struct PullMetadata {
    state: String,
    base: RefMetadata,
    head: RefMetadata,
}

#[derive(Deserialize)]
struct RefMetadata {
    sha: String,
    repo: Option<RepositoryMetadata>,
}

#[derive(Deserialize)]
struct RepositoryMetadata {
    full_name: String,
}

#[derive(Deserialize)]
struct ApiChangedFile {
    filename: String,
    status: String,
    patch: Option<String>,
}

#[derive(Deserialize)]
struct ApiComment {
    id: u64,
    body: Option<String>,
    user: Option<ApiUser>,
}

#[derive(Deserialize)]
struct ApiUser {
    login: String,
}

#[derive(Deserialize)]
struct ViewerData {
    viewer: ApiUser,
}

#[derive(Deserialize)]
struct GraphQlResponse<T> {
    data: Option<T>,
    errors: Option<Vec<Value>>,
}

#[derive(Deserialize)]
struct ThreadsData {
    repository: Option<ThreadsRepository>,
}

#[derive(Deserialize)]
struct ThreadsRepository {
    #[serde(rename = "pullRequest")]
    pull_request: Option<PullRequestThreads>,
}

#[derive(Deserialize)]
struct PullRequestThreads {
    #[serde(rename = "reviewThreads")]
    review_threads: ThreadConnection,
}

#[derive(Deserialize)]
struct ThreadConnection {
    nodes: Vec<ThreadNode>,
    #[serde(rename = "pageInfo")]
    page_info: PageInfo,
}

#[derive(Deserialize)]
struct PageInfo {
    #[serde(rename = "hasNextPage")]
    has_next_page: bool,
    #[serde(rename = "endCursor")]
    end_cursor: Option<String>,
}

#[derive(Deserialize)]
struct ThreadNode {
    id: String,
    #[serde(rename = "isResolved")]
    is_resolved: bool,
    #[serde(rename = "isOutdated")]
    is_outdated: bool,
    path: Option<String>,
    line: Option<usize>,
    comments: ThreadComments,
}

#[derive(Deserialize)]
struct ThreadComments {
    nodes: Vec<ThreadComment>,
}

#[derive(Deserialize)]
struct ThreadComment {
    #[serde(rename = "databaseId")]
    database_id: Option<u64>,
    body: Option<String>,
    author: Option<ApiUser>,
}

#[derive(Serialize)]
struct GraphQlRequest<'a> {
    query: &'a str,
    variables: Value,
}

impl GitHubClient {
    pub fn new(api_base: Url, token: String) -> Result<Self> {
        validate_api_base(&api_base)?;
        if token.trim().is_empty() || token.chars().any(char::is_control) {
            bail!("GitHub token отсутствует или содержит управляющие символы.");
        }
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|_| anyhow::anyhow!("Не удалось настроить HTTP-клиент GitHub."))?;
        Ok(Self {
            api_base,
            token,
            http,
        })
    }

    pub async fn fetch_snapshot(&self, identity: &PullRequestId) -> Result<PullRequestSnapshot> {
        identity.validate()?;
        let metadata = self.pull_metadata(identity).await?;
        validate_metadata(identity, &metadata)?;
        let files = self.changed_files(identity).await?;
        Ok(PullRequestSnapshot {
            base_sha: metadata.base.sha,
            head_sha: metadata.head.sha,
            files,
        })
    }

    pub async fn assert_current_head(&self, identity: &PullRequestId) -> Result<()> {
        identity.validate()?;
        let metadata = self.pull_metadata(identity).await?;
        validate_metadata(identity, &metadata)
    }

    pub async fn viewer_login(&self) -> Result<String> {
        let request = GraphQlRequest {
            query: VIEWER_QUERY,
            variables: json!({}),
        };
        let response: GraphQlResponse<ViewerData> =
            self.post_graphql(self.url(&["graphql"])?, &request).await?;
        if response
            .errors
            .as_ref()
            .is_some_and(|errors| !errors.is_empty())
        {
            bail!("GitHub GraphQL не смог определить viewer login.");
        }
        let login = response
            .data
            .context("GitHub GraphQL не вернул viewer login.")?
            .viewer
            .login;
        if login.trim().is_empty() || login.chars().any(char::is_control) {
            bail!("GitHub GraphQL вернул некорректный viewer login.");
        }
        Ok(login)
    }

    pub async fn existing_threads(
        &self,
        identity: &PullRequestId,
        viewer_login: &str,
    ) -> Result<Vec<ExistingThread>> {
        identity.validate()?;
        let mut threads = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let variables = json!({
                "owner": identity.owner,
                "name": identity.repository,
                "number": identity.number,
                "after": cursor,
            });
            let request = GraphQlRequest {
                query: THREADS_QUERY,
                variables,
            };
            let response: GraphQlResponse<ThreadsData> =
                self.post_graphql(self.url(&["graphql"])?, &request).await?;
            if response
                .errors
                .as_ref()
                .is_some_and(|errors| !errors.is_empty())
            {
                bail!("GitHub GraphQL не смог загрузить review threads.");
            }
            let data = response
                .data
                .context("GitHub GraphQL не вернул review threads.")?;
            let connection = data
                .repository
                .and_then(|repository| repository.pull_request)
                .map(|pull_request| pull_request.review_threads)
                .context("Pull request не найден в GitHub GraphQL.")?;
            for thread in connection.nodes {
                let Some(comment) = thread.comments.nodes.first() else {
                    continue;
                };
                if comment.author.as_ref().map(|author| author.login.as_str()) != Some(viewer_login)
                {
                    continue;
                }
                let Some(body) = comment.body.as_deref() else {
                    continue;
                };
                let Some(fingerprint) = parse_finding_fingerprint(body) else {
                    continue;
                };
                if body.len() > MAX_FINDING_COMMENT_BYTES {
                    bail!("Assistant review comment exceeds 16 KiB; review stopped.");
                }
                let (Some(comment_id), Some(path)) = (comment.database_id, thread.path.as_ref())
                else {
                    continue;
                };
                threads.push(ExistingThread {
                    thread_id: thread.id,
                    comment_id,
                    fingerprint: fingerprint.to_owned(),
                    path: path.clone(),
                    line: thread.line,
                    body: body.to_owned(),
                    is_resolved: thread.is_resolved,
                    is_outdated: thread.is_outdated,
                });
                if threads.len() > MAX_REVIEW_THREADS {
                    bail!("PR содержит больше 1000 review threads; обработка остановлена.");
                }
            }
            if !connection.page_info.has_next_page {
                break;
            }
            cursor = Some(
                connection
                    .page_info
                    .end_cursor
                    .context("GitHub GraphQL не вернул cursor для следующей страницы.")?,
            );
        }
        Ok(threads)
    }

    pub async fn existing_summary(
        &self,
        identity: &PullRequestId,
        viewer_login: &str,
    ) -> Result<Option<ExistingSummary>> {
        identity.validate()?;
        for page in 1..=MAX_ISSUE_COMMENT_PAGES {
            let url = self.issue_comments_url(identity, page)?;
            let comments: Vec<ApiComment> = self.get_json(url).await?;
            for comment in &comments {
                if comment.user.as_ref().map(|user| user.login.as_str()) != Some(viewer_login) {
                    continue;
                }
                let Some(body) = comment.body.as_deref() else {
                    continue;
                };
                if has_summary_marker(body) {
                    return Ok(Some(ExistingSummary {
                        comment_id: comment.id,
                    }));
                }
            }
            if comments.len() < PAGE_SIZE as usize {
                return Ok(None);
            }
        }
        bail!("Слишком много issue comments для ограниченного поиска summary marker.");
    }

    pub async fn upsert_summary(
        &self,
        identity: &PullRequestId,
        existing: Option<&ExistingSummary>,
        body: &str,
    ) -> Result<()> {
        self.assert_current_head(identity).await?;
        let payload = json!({ "body": body });
        match existing {
            Some(comment) => {
                let url = self.url(&[
                    "repos",
                    &identity.owner,
                    &identity.repository,
                    "issues",
                    "comments",
                    &comment.comment_id.to_string(),
                ])?;
                let _: Value = self.patch_json(url, &payload).await?;
            }
            None => {
                let url = self.url(&[
                    "repos",
                    &identity.owner,
                    &identity.repository,
                    "issues",
                    &identity.number.to_string(),
                    "comments",
                ])?;
                let _: Value = self.post_json(url, &payload).await?;
            }
        }
        Ok(())
    }

    pub async fn upsert_inline_comment(
        &self,
        identity: &PullRequestId,
        existing: Option<&ExistingThread>,
        head_sha: &str,
        path: &str,
        line: usize,
        body: &str,
    ) -> Result<()> {
        self.assert_current_head(identity).await?;
        if !valid_sha(head_sha) || line == 0 {
            bail!("Inline review comment требует актуальный head SHA и line.");
        }
        if let Some(thread) = existing {
            let url = self.url(&[
                "repos",
                &identity.owner,
                &identity.repository,
                "pulls",
                "comments",
                &thread.comment_id.to_string(),
            ])?;
            let _: Value = self.patch_json(url, &json!({ "body": body })).await?;
        } else {
            let url = self.url(&[
                "repos",
                &identity.owner,
                &identity.repository,
                "pulls",
                &identity.number.to_string(),
                "comments",
            ])?;
            let payload = json!({
                "body": body,
                "commit_id": head_sha,
                "path": path,
                "line": line,
                "side": "RIGHT"
            });
            let _: Value = self.post_json(url, &payload).await?;
        }
        Ok(())
    }

    pub async fn resolve_thread(&self, identity: &PullRequestId, thread_id: &str) -> Result<()> {
        self.assert_current_head(identity).await?;
        let request = GraphQlRequest {
            query: RESOLVE_THREAD_MUTATION,
            variables: json!({ "threadId": thread_id }),
        };
        let response: GraphQlResponse<Value> =
            self.post_graphql(self.url(&["graphql"])?, &request).await?;
        if response
            .errors
            .as_ref()
            .is_some_and(|errors| !errors.is_empty())
            || response
                .data
                .as_ref()
                .and_then(|data| data.pointer("/resolveReviewThread/thread/isResolved"))
                != Some(&Value::Bool(true))
        {
            bail!("GitHub не подтвердил resolve review thread.");
        }
        Ok(())
    }

    async fn pull_metadata(&self, identity: &PullRequestId) -> Result<PullMetadata> {
        self.get_json(self.url(&[
            "repos",
            &identity.owner,
            &identity.repository,
            "pulls",
            &identity.number.to_string(),
        ])?)
        .await
    }

    async fn changed_files(&self, identity: &PullRequestId) -> Result<Vec<ChangedFile>> {
        let mut files = Vec::new();
        let mut patch_bytes = 0usize;
        for page in 1..=31_u64 {
            let mut url = self.url(&[
                "repos",
                &identity.owner,
                &identity.repository,
                "pulls",
                &identity.number.to_string(),
                "files",
            ])?;
            url.query_pairs_mut()
                .append_pair("per_page", &PAGE_SIZE.to_string())
                .append_pair("page", &page.to_string());
            let page_files: Vec<ApiChangedFile> = self.get_json(url).await?;
            let page_len = page_files.len();
            if page_len == 0 {
                return Ok(files);
            }
            if files.len().saturating_add(page_len) > MAX_CHANGED_FILES {
                bail!("Pull request содержит больше 3000 changed files; review не запущен.");
            }
            for file in page_files {
                if let Some(patch) = &file.patch {
                    patch_bytes = patch_bytes.saturating_add(patch.len());
                    if patch_bytes > MAX_DIFF_BYTES {
                        bail!("Pull request text diff превышает 8 MiB; review не запущен.");
                    }
                }
                let added_lines = added_line_numbers(file.patch.as_deref());
                files.push(ChangedFile {
                    path: file.filename,
                    status: file.status,
                    patch: file.patch,
                    added_lines,
                });
            }
            if page_len < PAGE_SIZE as usize {
                return Ok(files);
            }
        }
        bail!("Не удалось завершить pagination changed files.")
    }

    fn issue_comments_url(&self, identity: &PullRequestId, page: u64) -> Result<Url> {
        let mut url = self.url(&[
            "repos",
            &identity.owner,
            &identity.repository,
            "issues",
            &identity.number.to_string(),
            "comments",
        ])?;
        url.query_pairs_mut()
            .append_pair("per_page", &PAGE_SIZE.to_string())
            .append_pair("page", &page.to_string());
        Ok(url)
    }

    fn url(&self, segments: &[&str]) -> Result<Url> {
        let mut url = self.api_base.clone();
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("GitHub API base URL не поддерживает path segments."))?;
        path.pop_if_empty();
        for segment in segments {
            path.push(segment);
        }
        drop(path);
        Ok(url)
    }

    async fn get_json<T: for<'de> Deserialize<'de>>(&self, url: Url) -> Result<T> {
        let response = self
            .request(self.http.get(url))
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("Не удалось подключиться к GitHub API."))?;
        self.decode_response(response).await
    }

    async fn post_json<T: for<'de> Deserialize<'de>>(
        &self,
        url: Url,
        payload: &Value,
    ) -> Result<T> {
        let response = self
            .request(self.http.post(url).json(payload))
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("Не удалось отправить запрос GitHub API."))?;
        self.decode_response(response).await
    }

    async fn patch_json<T: for<'de> Deserialize<'de>>(
        &self,
        url: Url,
        payload: &Value,
    ) -> Result<T> {
        let response = self
            .request(self.http.patch(url).json(payload))
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("Не удалось отправить запрос GitHub API."))?;
        self.decode_response(response).await
    }

    async fn post_graphql<T: for<'de> Deserialize<'de>>(
        &self,
        url: Url,
        payload: &GraphQlRequest<'_>,
    ) -> Result<T> {
        let response = self
            .request(self.http.post(url).json(payload))
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("Не удалось выполнить GitHub GraphQL request."))?;
        self.decode_response(response).await
    }

    fn request(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request
            .bearer_auth(&self.token)
            .header("User-Agent", "developer-assistant-review")
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
    }

    async fn decode_response<T: for<'de> Deserialize<'de>>(&self, response: Response) -> Result<T> {
        if !response.status().is_success() {
            bail!("GitHub API вернул HTTP {}.", response.status());
        }
        let body = bounded_body(response, MAX_RESPONSE_BYTES).await?;
        serde_json::from_slice(&body).context("GitHub API вернул некорректный JSON.")
    }
}

fn validate_metadata(identity: &PullRequestId, metadata: &PullMetadata) -> Result<()> {
    if metadata.state != "open" {
        bail!("Pull request закрыт; review не публикуется.");
    }
    let repository = identity.full_name();
    let base_repo = metadata
        .base
        .repo
        .as_ref()
        .map(|repo| repo.full_name.as_str())
        .context("GitHub API не вернул base repository.")?;
    let head_repo = metadata
        .head
        .repo
        .as_ref()
        .map(|repo| repo.full_name.as_str())
        .context("Fork или удалённый head repository не поддерживается.")?;
    if !same_repository(base_repo, &repository) || !same_repository(head_repo, &repository) {
        bail!("Fork pull request не поддерживается.");
    }
    if metadata.base.sha != identity.expected_base_sha
        || metadata.head.sha != identity.expected_head_sha
    {
        bail!("Pull request base/head изменился; устаревший review не публикуется.");
    }
    Ok(())
}

fn same_repository(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

fn valid_repository_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_api_base(url: &Url) -> Result<()> {
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("GitHub API base URL не должен содержать credentials, query или fragment.");
    }
    let host = url
        .host_str()
        .context("GitHub API base URL должен содержать host.")?;
    let loopback = host.eq_ignore_ascii_case("localhost")
        || IpAddr::from_str(host.trim_matches(['[', ']']))
            .is_ok_and(|address| address.is_loopback());
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        bail!("Удалённый GitHub API endpoint должен использовать HTTPS.");
    }
    Ok(())
}

fn added_line_numbers(patch: Option<&str>) -> Vec<usize> {
    let Some(patch) = patch else {
        return Vec::new();
    };
    let mut added = Vec::new();
    let mut new_line = None;
    for line in patch.lines() {
        if line.starts_with("@@") {
            new_line = parse_new_line_start(line);
        } else if line.starts_with('+') {
            if let Some(current) = new_line {
                added.push(current);
                new_line = current.checked_add(1);
            }
        } else if line.starts_with(' ') {
            new_line = new_line.and_then(|current| current.checked_add(1));
        }
    }
    added.sort_unstable();
    added.dedup();
    added
}

fn parse_new_line_start(header: &str) -> Option<usize> {
    let plus = header.find('+')?;
    let range = header.get(plus + 1..)?;
    let number = range
        .split(|character: char| !character.is_ascii_digit())
        .next()?;
    number.parse().ok()
}

async fn bounded_body(mut response: Response, limit: usize) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        bail!("Ответ GitHub API превышает допустимый размер.");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| anyhow::anyhow!("Не удалось прочитать ответ GitHub API."))?
    {
        if body.len().saturating_add(chunk.len()) > limit {
            bail!("Ответ GitHub API превышает допустимый размер.");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::{
        PullMetadata, PullRequestId, added_line_numbers, parse_new_line_start, validate_metadata,
    };
    use serde_json::json;

    #[test]
    fn only_added_lines_from_hunks_are_eligible_for_inline_anchors() {
        let patch = "@@ -2,3 +2,4 @@\n before\n-old\n+new one\n+new two\n after\n@@ -9,1 +10,1 @@\n-previous\n+replacement\n";
        assert_eq!(added_line_numbers(Some(patch)), vec![3, 4, 10]);
        assert!(added_line_numbers(None).is_empty());
    }

    #[test]
    fn added_prefix_increment_is_counted_as_source_not_file_header() {
        let patch = "@@ -0,0 +1,2 @@\n+++counter;\n+next();\n";
        assert_eq!(added_line_numbers(Some(patch)), vec![1, 2]);
    }

    #[test]
    fn malformed_hunk_does_not_create_inline_anchor() {
        assert_eq!(parse_new_line_start("@@ malformed @@"), None);
        assert!(added_line_numbers(Some("@@ malformed @@\n+unanchored")).is_empty());
    }

    #[test]
    fn metadata_rejects_fork_and_stale_base_or_head() {
        let identity = PullRequestId {
            owner: "acme".to_owned(),
            repository: "transaction-import".to_owned(),
            number: 7,
            expected_base_sha: "b".repeat(40),
            expected_head_sha: "a".repeat(40),
        };
        let matching = metadata("acme/transaction-import", "b", "a");
        assert!(validate_metadata(&identity, &matching).is_ok());

        let fork = metadata("attacker/fork", "b", "a");
        assert!(validate_metadata(&identity, &fork).is_err());
        let stale_base = metadata("acme/transaction-import", "c", "a");
        assert!(validate_metadata(&identity, &stale_base).is_err());
        let stale_head = metadata("acme/transaction-import", "b", "c");
        assert!(validate_metadata(&identity, &stale_head).is_err());
    }

    fn metadata(head_repository: &str, base_sha: &str, head_sha: &str) -> PullMetadata {
        serde_json::from_value(json!({
            "state": "open",
            "base": {
                "sha": base_sha.repeat(40),
                "repo": {"full_name":"acme/transaction-import"}
            },
            "head": {
                "sha": head_sha.repeat(40),
                "repo": {"full_name":head_repository}
            }
        }))
        .expect("synthetic pull metadata deserializes")
    }
}
