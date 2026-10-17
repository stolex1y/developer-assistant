use crate::review::corpus::contains_secret_like_value;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const SUMMARY_MARKER: &str = "<!-- developer-assistant:summary:v1 -->";
const FINDING_MARKER_PREFIX: &str = "<!-- developer-assistant:finding:v1:";
const MAX_FINDINGS: usize = 32;
const MAX_SUMMARY_BYTES: usize = 8 * 1024;
const MAX_RENDERED_SUMMARY_BYTES: usize = 64 * 1024;
const MAX_DETAIL_BYTES: usize = 2 * 1024;
const MAX_EVIDENCE_BYTES: usize = 1024;
const MAX_INLINE_FENCE: usize = 16;
const SUMMARY_DISCLAIMER: &str = "AI-резюме не проверяется отдельно. У findings сверяются path/evidence с diff; смысл замечаний требует ручной оценки.";

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewReport {
    pub summary: String,
    pub findings: Vec<ReviewFinding>,
    pub resolved_findings: Vec<ResolvedFinding>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewFinding {
    pub category: FindingCategory,
    pub title: String,
    pub detail: String,
    pub path: String,
    pub line: Option<usize>,
    pub evidence: String,
    #[serde(skip)]
    pub removed_file: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FindingCategory {
    Bug,
    Architecture,
    Recommendation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResolvedFinding {
    pub fingerprint: String,
    pub path: String,
    pub evidence: String,
}

impl ReviewReport {
    pub fn parse(content: &str) -> Result<Self> {
        let report: Self = serde_json::from_str(content)
            .context("DeepSeek review JSON не соответствует report contract.")?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        validate_text(&self.summary, MAX_SUMMARY_BYTES, "summary")?;
        if self.findings.len() > MAX_FINDINGS {
            bail!("DeepSeek вернул больше 32 findings.");
        }

        let mut fingerprints = HashSet::with_capacity(self.findings.len());
        for finding in &self.findings {
            validate_text(&finding.title, 200, "finding title")?;
            if finding
                .title
                .chars()
                .any(|character| matches!(character, '\n' | '\r' | '\t'))
            {
                bail!("Finding title должен занимать одну строку.");
            }
            validate_text(&finding.detail, MAX_DETAIL_BYTES, "finding detail")?;
            validate_text(&finding.path, 1024, "finding path")?;
            validate_text(&finding.evidence, MAX_EVIDENCE_BYTES, "finding evidence")?;
            if finding.line == Some(0) {
                bail!("Finding line должен начинаться с 1.");
            }
            let fingerprint = finding_fingerprint(finding);
            if !fingerprints.insert(fingerprint) {
                bail!("DeepSeek вернул повторяющийся finding fingerprint.");
            }
        }

        if self.resolved_findings.len() > 100 {
            bail!("DeepSeek вернул слишком много resolved findings.");
        }
        let mut resolved_fingerprints = HashSet::with_capacity(self.resolved_findings.len());
        for resolved in &self.resolved_findings {
            validate_fingerprint(&resolved.fingerprint)?;
            if !resolved_fingerprints.insert(resolved.fingerprint.as_str()) {
                bail!("DeepSeek вернул повторяющийся resolved fingerprint.");
            }
            validate_text(&resolved.path, 1024, "resolved finding path")?;
            validate_text(
                &resolved.evidence,
                MAX_EVIDENCE_BYTES,
                "resolution evidence",
            )?;
        }
        Ok(())
    }
}

impl FindingCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bug => "bug",
            Self::Architecture => "architecture",
            Self::Recommendation => "recommendation",
        }
    }

    fn heading(&self) -> &'static str {
        match self {
            Self::Bug => "Потенциальные баги",
            Self::Architecture => "Архитектурные проблемы",
            Self::Recommendation => "Рекомендации",
        }
    }
}

pub fn finding_fingerprint(finding: &ReviewFinding) -> String {
    let normalized_title = finding
        .title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let material = format!(
        "{}\0{}\0{}",
        finding.category.as_str(),
        finding.path,
        normalized_title.to_lowercase()
    );
    let digest = Sha256::digest(material.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn parse_finding_fingerprint(body: &str) -> Option<&str> {
    let marker = body.lines().next()?.strip_prefix(FINDING_MARKER_PREFIX)?;
    let fingerprint = marker.strip_suffix(" -->")?;
    if is_valid_fingerprint(fingerprint) {
        Some(fingerprint)
    } else {
        None
    }
}

pub fn has_summary_marker(body: &str) -> bool {
    body.lines().next() == Some(SUMMARY_MARKER)
}

pub fn render_finding_comment(finding: &ReviewFinding) -> Result<String> {
    if finding.removed_file {
        bail!("Findings об удалённых файлах публикуются только в PR summary.");
    }
    let fingerprint = finding_fingerprint(finding);
    let evidence_fence = code_fence(&finding.evidence)?;
    let line = finding
        .line
        .map(|line| format!(":{}", line))
        .unwrap_or_default();
    Ok(format!(
        "{FINDING_MARKER_PREFIX}{fingerprint} -->\n**[{}] {} — `{}`{}**\n\n{}\n\nEvidence:\n{evidence_fence}\n{}\n{evidence_fence}",
        markdown_escape(finding.category.as_str()),
        markdown_escape(&finding.title),
        markdown_escape(&finding.path),
        line,
        markdown_escape(&finding.detail),
        finding.evidence
    ))
}

pub fn render_summary(report: &ReviewReport, head_sha: &str) -> Result<String> {
    let mut output = format!(
        "{SUMMARY_MARKER}\n\n# AI code review\n\n**Head:** `{}`\n\n> **Ограничение:** {}\n\n{}\n",
        markdown_escape(head_sha),
        SUMMARY_DISCLAIMER,
        markdown_escape(&report.summary)
    );
    for category in [
        FindingCategory::Bug,
        FindingCategory::Architecture,
        FindingCategory::Recommendation,
    ] {
        output.push_str("\n## ");
        output.push_str(category.heading());
        output.push('\n');
        let mut found = false;
        for finding in report
            .findings
            .iter()
            .filter(|finding| finding.category == category)
        {
            found = true;
            output.push_str("\n### ");
            output.push_str(&markdown_escape(&finding.title));
            output.push_str("\n\n**Файл:** `");
            output.push_str(&markdown_escape(&finding.path));
            if let Some(line) = finding.line {
                output.push(':');
                output.push_str(&line.to_string());
            }
            output.push_str("`\n\n");
            if finding.removed_file {
                output.push_str(
                    "**Сторона diff:** удалённый файл; evidence относится к удалённому коду и описывает возможную регрессию от удаления, не текущее содержимое.\n\n",
                );
            }
            output.push_str(&markdown_escape(&finding.detail));
            output.push_str("\n\nEvidence:\n");
            let fence = code_fence(&finding.evidence)?;
            output.push_str(&fence);
            output.push('\n');
            output.push_str(&finding.evidence);
            output.push('\n');
            output.push_str(&fence);
            output.push('\n');
        }
        if !found {
            output.push_str("\nНе обнаружено в проверенной части diff.\n");
        }
    }
    if output.len() > MAX_RENDERED_SUMMARY_BYTES {
        bail!("Сформированный PR summary превышает допустимый размер.");
    }
    Ok(output)
}

fn validate_text(value: &str, max_bytes: usize, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > max_bytes {
        bail!("DeepSeek вернул пустое или слишком длинное поле {field}.");
    }
    if value
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
        || value.contains("<!-- developer-assistant:")
        || contains_secret_like_value(value)
    {
        bail!("DeepSeek вернул небезопасное поле {field}.");
    }
    Ok(())
}

fn validate_fingerprint(value: &str) -> Result<()> {
    if !is_valid_fingerprint(value) {
        bail!("DeepSeek вернул некорректный finding fingerprint.");
    }
    Ok(())
}

fn is_valid_fingerprint(value: &str) -> bool {
    value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn code_fence(value: &str) -> Result<String> {
    let max_run = value
        .lines()
        .map(|line| {
            line.trim_start_matches([' ', '\t'])
                .chars()
                .take_while(|character| *character == '`')
                .count()
        })
        .max()
        .unwrap_or(0);
    let fence_length = 3.max(max_run + 1);
    if fence_length > MAX_INLINE_FENCE {
        bail!("Evidence содержит слишком длинную последовательность backticks.");
    }
    Ok("`".repeat(fence_length))
}

fn markdown_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '\\' | '`'
                | '*'
                | '_'
                | '{'
                | '}'
                | '['
                | ']'
                | '('
                | ')'
                | '#'
                | '+'
                | '-'
                | '.'
                | '!'
                | '|'
                | '<'
                | '>'
                | '&'
                | '@'
        ) {
            output.push('\\');
        }
        output.push(character);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{ReviewFinding, ReviewReport, markdown_escape};
    use serde_json::json;

    #[test]
    fn escaped_comment_text_cannot_create_user_mentions() {
        assert_eq!(markdown_escape("@reviewer"), "\\@reviewer");
    }

    #[test]
    fn finding_title_is_single_line() {
        let report: ReviewReport = serde_json::from_value(json!({
            "summary":"Review complete.",
            "findings":[{
                "category":"bug",
                "title":"Injected title\n@reviewer",
                "detail":"A synthetic detail.",
                "path":"src/file.rs",
                "line":1,
                "evidence":"safe evidence"
            }],
            "resolved_findings":[]
        }))
        .expect("report schema deserializes");
        assert!(report.validate().is_err());
    }

    #[test]
    fn removed_file_findings_are_labeled_and_summary_only() {
        let finding = ReviewFinding {
            category: super::FindingCategory::Bug,
            title: "Validation removed".to_owned(),
            detail: "The deletion may bypass validation.".to_owned(),
            path: "src/deleted.rs".to_owned(),
            line: None,
            evidence: "old_validate();".to_owned(),
            removed_file: true,
        };
        assert!(super::render_finding_comment(&finding).is_err());

        let report = ReviewReport {
            summary: "Review complete.".to_owned(),
            findings: vec![finding],
            resolved_findings: Vec::new(),
        };
        let head_sha = "a".repeat(40);
        let summary = super::render_summary(&report, &head_sha)
            .expect("removed-file finding renders in summary");
        assert!(
            summary.contains(
                "**Сторона diff:** удалённый файл; evidence относится к удалённому коду и описывает возможную регрессию от удаления, не текущее содержимое."
            )
        );
        assert!(summary.contains(r"**Файл:** `src/deleted\.rs`"));
        assert!(!summary.contains(r"src/deleted\.rs:"));
    }
}
