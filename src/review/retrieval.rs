use crate::review::corpus::SourceDocument;
use anyhow::{Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::LazyLock};

pub const MAX_PASSAGE_BYTES: usize = 1_200;
pub const MAX_SEARCH_RESULTS: usize = 8;
const MAX_CANDIDATES_PER_RANKER: usize = 32;
const RRF_CONSTANT: f64 = 60.0;
const MIN_SEMANTIC_SIMILARITY: f64 = 0.15;

static WORD_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?u)[\p{L}\p{N}_]+").expect("fixed word pattern is valid"));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Passage {
    pub source_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub passage_index: usize,
    pub score: f64,
}

struct TermStats {
    frequencies: HashMap<String, usize>,
    length: usize,
}

pub struct HybridRetriever {
    term_stats: Vec<TermStats>,
    document_frequencies: HashMap<String, usize>,
    average_document_length: f64,
}

impl HybridRetriever {
    pub fn new(passages: &[Passage]) -> Self {
        let term_stats = passages
            .iter()
            .map(|passage| {
                let mut frequencies = HashMap::new();
                for token in tokenize(&format!("{} {}", passage.source_path, passage.text)) {
                    *frequencies.entry(token).or_insert(0) += 1;
                }
                let length = frequencies.values().sum();
                TermStats {
                    frequencies,
                    length,
                }
            })
            .collect::<Vec<_>>();

        let mut document_frequencies = HashMap::new();
        for stats in &term_stats {
            for token in stats.frequencies.keys() {
                *document_frequencies.entry(token.clone()).or_insert(0) += 1;
            }
        }
        let average_document_length = if term_stats.is_empty() {
            1.0
        } else {
            (term_stats.iter().map(|stats| stats.length).sum::<usize>() as f64
                / term_stats.len() as f64)
                .max(1.0)
        };

        Self {
            term_stats,
            document_frequencies,
            average_document_length,
        }
    }

    pub fn search(
        &self,
        passages: &[Passage],
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        if limit == 0 || limit > MAX_SEARCH_RESULTS {
            bail!("Число результатов search_code должно быть от 1 до {MAX_SEARCH_RESULTS}.");
        }
        if passages.len() != self.term_stats.len() {
            bail!("Hybrid retrieval index не соответствует passage corpus.");
        }
        if query.trim().is_empty() || passages.is_empty() {
            return Ok(Vec::new());
        }

        let query_tokens = tokenize(query);
        let mut semantic = passages
            .iter()
            .enumerate()
            .filter_map(|(index, passage)| {
                cosine_similarity(query_embedding, &passage.embedding)
                    .filter(|score| *score >= MIN_SEMANTIC_SIMILARITY)
                    .map(|score| (index, score))
            })
            .collect::<Vec<_>>();
        semantic.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| passage_order(&passages[*left_index], &passages[*right_index]))
        });
        semantic.truncate(MAX_CANDIDATES_PER_RANKER);

        let mut lexical = self
            .term_stats
            .iter()
            .enumerate()
            .filter_map(|(index, stats)| {
                let score = self.bm25_score(stats, &query_tokens, passages.len());
                (score > 0.0).then_some((index, score))
            })
            .collect::<Vec<_>>();
        lexical.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| passage_order(&passages[*left_index], &passages[*right_index]))
        });
        lexical.truncate(MAX_CANDIDATES_PER_RANKER);

        let mut fused_scores = vec![0.0; passages.len()];
        add_reciprocal_ranks(&mut fused_scores, &semantic);
        add_reciprocal_ranks(&mut fused_scores, &lexical);
        let mut ranked = fused_scores
            .into_iter()
            .enumerate()
            .filter_map(|(passage_index, score)| {
                (score > 0.0).then_some(SearchHit {
                    passage_index,
                    score,
                })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right.score.total_cmp(&left.score).then_with(|| {
                passage_order(
                    &passages[left.passage_index],
                    &passages[right.passage_index],
                )
            })
        });
        ranked.truncate(limit);
        Ok(ranked)
    }

    fn bm25_score(&self, stats: &TermStats, query_tokens: &[String], document_count: usize) -> f64 {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let document_length = stats.length as f64;
        let mut score = 0.0;
        for token in query_tokens {
            let Some(term_frequency) = stats.frequencies.get(token) else {
                continue;
            };
            let document_frequency = self.document_frequencies[token] as f64;
            let document_count = document_count as f64;
            let inverse_document_frequency = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            let term_frequency = *term_frequency as f64;
            let denominator = term_frequency
                + K1 * (1.0 - B + B * document_length / self.average_document_length);
            score += inverse_document_frequency * term_frequency * (K1 + 1.0) / denominator;
        }
        score
    }
}

pub fn chunk_documents(documents: &[SourceDocument]) -> Vec<Passage> {
    let mut passages = Vec::new();
    for document in documents {
        let mut text = String::new();
        let mut start_line = 1usize;
        let mut end_line = 0usize;

        for (index, line) in document.text.lines().enumerate() {
            let line_number = index + 1;
            if line.len() > MAX_PASSAGE_BYTES {
                flush_passage(
                    &document.path,
                    &mut text,
                    start_line,
                    end_line,
                    &mut passages,
                );
                for fragment in split_at_utf8_boundary(line, MAX_PASSAGE_BYTES) {
                    passages.push(Passage {
                        source_path: document.path.clone(),
                        start_line: line_number,
                        end_line: line_number,
                        text: fragment.to_owned(),
                        embedding: Vec::new(),
                    });
                }
                start_line = line_number + 1;
                end_line = line_number;
                continue;
            }

            if !text.is_empty() && text.len() + line.len() + 1 > MAX_PASSAGE_BYTES {
                flush_passage(
                    &document.path,
                    &mut text,
                    start_line,
                    end_line,
                    &mut passages,
                );
                start_line = line_number;
            }
            if text.is_empty() {
                start_line = line_number;
            }
            text.push_str(line);
            text.push('\n');
            end_line = line_number;
        }
        flush_passage(
            &document.path,
            &mut text,
            start_line,
            end_line,
            &mut passages,
        );
    }
    passages
}

fn flush_passage(
    path: &str,
    text: &mut String,
    start_line: usize,
    end_line: usize,
    passages: &mut Vec<Passage>,
) {
    if text.is_empty() {
        return;
    }
    passages.push(Passage {
        source_path: path.to_owned(),
        start_line,
        end_line,
        text: std::mem::take(text),
        embedding: Vec::new(),
    });
}

fn split_at_utf8_boundary(text: &str, max_bytes: usize) -> Vec<&str> {
    let mut fragments = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let mut end = remaining.len().min(max_bytes);
        while !remaining.is_char_boundary(end) {
            end -= 1;
        }
        if end == 0 {
            break;
        }
        fragments.push(&remaining[..end]);
        remaining = &remaining[end..];
    }
    fragments
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for matched in WORD_PATTERN.find_iter(text) {
        let word = matched.as_str();
        if word.chars().count() < 2 {
            continue;
        }
        let full_identifier = word.to_lowercase();
        tokens.push(full_identifier.clone());
        let mut parts = Vec::new();
        split_identifier(word, &mut parts);
        tokens.extend(parts.into_iter().filter(|part| part != &full_identifier));
    }
    tokens
}

fn split_identifier(word: &str, output: &mut Vec<String>) {
    let chars = word.chars().collect::<Vec<_>>();
    let mut start = 0usize;
    for index in 1..chars.len() {
        let previous = chars[index - 1];
        let current = chars[index];
        let next_is_lowercase = chars
            .get(index + 1)
            .is_some_and(|character| character.is_lowercase());
        let boundary = current == '_'
            || (current.is_uppercase()
                && (previous.is_lowercase()
                    || previous.is_numeric()
                    || (previous.is_uppercase() && next_is_lowercase)));
        if boundary {
            push_identifier_part(&chars[start..index], output);
            start = if current == '_' { index + 1 } else { index };
        }
    }
    push_identifier_part(&chars[start..], output);
}

fn push_identifier_part(chars: &[char], output: &mut Vec<String>) {
    if chars.len() > 1 {
        output.push(chars.iter().collect::<String>().to_lowercase());
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    Some(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

fn add_reciprocal_ranks(scores: &mut [f64], ranked: &[(usize, f64)]) {
    for (rank, (index, _)) in ranked.iter().enumerate() {
        scores[*index] += 1.0 / (RRF_CONSTANT + rank as f64 + 1.0);
    }
}

fn passage_order(left: &Passage, right: &Passage) -> std::cmp::Ordering {
    left.source_path
        .cmp(&right.source_path)
        .then_with(|| left.start_line.cmp(&right.start_line))
        .then_with(|| left.end_line.cmp(&right.end_line))
}
