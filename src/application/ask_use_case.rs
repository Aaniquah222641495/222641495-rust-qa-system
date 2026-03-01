use anyhow::Result;
use tokenizers::Tokenizer;

use crate::data::{chunker::Chunker, loader::DocxLoader, preprocessor::Preprocessor};
use crate::domain::traits::DocumentSource;
use crate::infra::{checkpoint::CheckpointManager, tokenizer_store::TokenizerStore};
use crate::ml::inferencer::Inferencer;


pub struct AskUseCase {
    checkpoint_dir: String,
    docs_dir:       String,
    tokenizer:      Tokenizer,
    inferencer:     Inferencer,
}

impl AskUseCase {
    pub fn new(checkpoint_dir: String, docs_dir: String) -> Result<Self> {
        let tok_store  = TokenizerStore::new(&checkpoint_dir);
        let tokenizer  = tok_store.load()?;
        let ckpt       = CheckpointManager::new(&checkpoint_dir);
        let inferencer = Inferencer::from_checkpoint(&ckpt, &tokenizer)?;
        Ok(Self { checkpoint_dir, docs_dir, tokenizer, inferencer })
    }

    pub fn answer(&self, question: &str) -> Result<String> {
        let loader   = DocxLoader::new(&self.docs_dir);
        let raw_docs = loader.load_all()?;
        if raw_docs.is_empty() {
            return Ok("No documents found.".to_string());
        }

        let prep    = Preprocessor::new();
        // 100-word overlap ensures month-header paragraphs always share a chunk
        // with the calendar events that follow them (~70 words later).
        let chunker = Chunker::new(200, 100);

        let tagged_chunks: Vec<(String, String)> = raw_docs
            .iter()
            .flat_map(|d| {
                let source = d.source.clone();
                chunker.chunk(&prep.clean(&d.text))
                    .into_iter()
                    .map(move |chunk| (source.clone(), chunk))
            })
            .collect();

        if tagged_chunks.is_empty() {
            return Ok("Could not extract text from documents.".to_string());
        }

        let question_year = extract_year(question);
        let filtered_chunks: Vec<(String, String)> = if let Some(year) = question_year {
            let year_filtered: Vec<_> = tagged_chunks.iter()
                .filter(|(source, _)| source.contains(year))
                .cloned()
                .collect();
            if year_filtered.is_empty() {
                tagged_chunks.clone()
            } else {
                year_filtered
            }
        } else {
            tagged_chunks.clone()
        };

        let top_chunks = rank_chunks(question, &filtered_chunks, 20);
        let mut best_confidence = 0.0f32;

        for (_source, context) in &top_chunks {
            match self.inferencer.predict(question, context, &self.tokenizer) {
                Ok((_answer, confidence)) => {
                    if confidence > best_confidence {
                        best_confidence = confidence;
                    }
                }
                Err(e) => tracing::warn!("Inference error: {e}"),
            }
        }

        tracing::info!("Best confidence score: {:.4}", best_confidence);

        // Search ALL filtered chunks (not just top-20) so formal calendar events
        // are never displaced by committee-meeting chunks in the keyword ranking.
        let mut best_answer = String::new();
        let mut best_answer_score = -1i32;

        for (_source, chunk) in &filtered_chunks {
            if let Some((answer, score)) = extract_clean_answer(question, chunk) {
                if score > best_answer_score {
                    best_answer_score = score;
                    best_answer = answer;
                }
            }
        }

        if !best_answer.is_empty() {
            return Ok(best_answer);
        }

        Ok("I don't know based on the documents.".to_string())
    }
}

fn extract_year(question: &str) -> Option<&str> {
    let bytes = question.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            let run = &question[start..i];
            if run.len() == 4
                && (run.starts_with("202") || run.starts_with("203"))
            {
                return Some(run);
            }
        } else {
            i += 1;
        }
    }
    None
}

fn rank_chunks(
    question: &str,
    chunks:   &[(String, String)],
    top_k:    usize,
) -> Vec<(String, String)> {
    let q_words: Vec<String> = question.split_whitespace()
        .filter(|w| w.len() > 3 || w.chars().all(|c| c.is_ascii_digit()))
        .map(|w| w.to_lowercase())
        .collect();

    let total_weight: f32 = q_words.iter().map(|w| w.len() as f32).sum::<f32>() + 1.0;
    let mut scored: Vec<(f32, &(String, String))> = chunks.iter().map(|pair| {
        let chunk_lower = pair.1.to_lowercase();
        let score = q_words.iter()
            .filter(|w| chunk_lower.contains(w.as_str()))
            .map(|w| w.len() as f32)
            .sum::<f32>() / total_weight;
        (score, pair)
    }).collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter()
        .take(top_k)
        .map(|(_, pair)| pair.clone())
        .collect()
}

/// Score pipe-delimited cells by keyword overlap, then reconstruct a date answer
/// by combining the winning cell with the day number from the preceding cell and
/// the month+year header found by scanning backward through the chunk.
fn extract_clean_answer(question: &str, chunk: &str) -> Option<(String, i32)> {
    let months     = ["january","february","march","april","may","june",
                      "july","august","september","october","november","december"];
    let months_cap = ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"];

    let stop_words = ["when","what","does","will","this","that","have",
                      "from","held","the","and","for","are","was",
                      "2024","2025","2026","2027"];

    let key_terms: Vec<String> = question.split_whitespace()
        .map(|w| w.to_lowercase()
            .trim_matches(|c: char| c.is_ascii_punctuation())
            .to_string())
        .filter(|w| !w.is_empty()
            && !stop_words.contains(&w.as_str())
            && (w.len() > 2 || w.chars().all(|c| c.is_ascii_digit())))
        .collect();

    if key_terms.is_empty() { return None; }

    let segments: Vec<&str> = chunk
        .split(['\n', '|'])
        .map(|s| s.trim())
        .filter(|s| s.len() > 2)
        .collect();

    let mut best_idx: Option<usize> = None;
    let mut best_score = 0i32;

    for (i, seg) in segments.iter().enumerate() {
        let sl = seg.to_lowercase();
        let mut score = 0i32;

        // Weight by term length so longer, specific terms outrank short generic ones.
        for term in &key_terms {
            if contains_word(&sl, term) {
                score += term.len() as i32;
            }
        }

        // Penalise bare day-number cells and very short cells.
        let alnum: String = seg.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if !alnum.is_empty() && alnum.chars().all(|c| c.is_ascii_digit()) { score -= 10; }
        if seg.len() < 5 { score -= 5; }

        // ALL-CAPS bonus: formal calendar events are all-caps; admin text is not.
        // Only applied when at least one key-term matched to avoid month headers winning.
        if score > 0 {
            let alpha: Vec<char> = seg.chars().filter(|c| c.is_alphabetic()).collect();
            if !alpha.is_empty() {
                let upper = alpha.iter().filter(|c| c.is_uppercase()).count();
                if upper as f32 / alpha.len() as f32 >= 0.7 {
                    score += 5;
                }
            }
        }

        // Penalise if a short numeric key-term (e.g. "2" in "Term 2") is absent.
        let numeric_terms: Vec<&str> = key_terms.iter()
            .filter(|t| t.chars().all(|c| c.is_ascii_digit()))
            .map(|t| t.as_str())
            .collect();
        if !numeric_terms.is_empty() {
            let seg_words: Vec<&str> = sl.split_whitespace().collect();
            let missing = numeric_terms.iter()
                .any(|n| !seg_words.contains(n));
            if missing { score -= 8; }
        }

        if score > best_score && score >= 3 {
            best_score = score;
            best_idx   = Some(i);
        }
    }

    let idx = best_idx?;

    // Tiebreaker: sum of key-term scores from all non-winning segments.
    // e.g. the December chunk beats April for "SUMMER GRADUATION" because it
    // also contains "END OF TERM 4 End of Year" giving it a higher context score.
    let context_score: i32 = segments.iter().enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, seg)| {
            let sl = seg.to_lowercase();
            key_terms.iter()
                .filter(|t| contains_word(&sl, t))
                .map(|t| t.len() as i32)
                .sum::<i32>()
        })
        .sum::<i32>()
        .min(999);

    let combined_score = best_score * 1000 + context_score;

    // Prefer a day number inside the winning cell; fall back to the last word of the
    // preceding cell (e.g. "… 29 | GOOD FRIDAY" where 29 is in the adjacent cell).
    let day_from_self: Option<String> = segments[idx]
        .split_whitespace()
        .find(|w| parse_day(w).is_some())
        .and_then(parse_day);

    let day_from_prev: Option<String> = if idx > 0 {
        segments[idx - 1]
            .split_whitespace()
            .next_back()
            .and_then(parse_day)
    } else {
        None
    };

    let day = day_from_self.or(day_from_prev);

    let month_year_from_seg = |seg: &str| -> Option<String> {
        let sl = seg.to_lowercase();
        let (_, cap) = months.iter().zip(months_cap.iter())
            .find(|(m, _)| sl.contains(*m))?;
        let words: Vec<&str> = seg.split_whitespace().collect();
        let yr = words.iter()
            .find(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
            .map(|w| w.to_string())
            // Fallback: join two adjacent digit tokens that form a 4-digit year
            // (handles "202 6" split across Word runs).
            .or_else(|| {
                words.windows(2).find_map(|pair| {
                    let joined = format!("{}{}", pair[0], pair[1]);
                    if joined.len() == 4
                        && joined.chars().all(|c| c.is_ascii_digit())
                        && (joined.starts_with("202") || joined.starts_with("203"))
                    {
                        Some(joined)
                    } else {
                        None
                    }
                })
            });
        Some(match yr {
            Some(y) => format!("{} {}", cap, y),
            None    => cap.to_string(),
        })
    };

    // Calendar headers sometimes split "DECEMBER" and "2026" into adjacent cells,
    // so when a month-only segment is found we also check its neighbours for the year.
    let find_year_near = |segs: &[&str], k: usize| -> Option<String> {
        let try_seg = |j: usize| -> Option<String> {
            segs.get(j).and_then(|s| {
                s.split_whitespace()
                    .find(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
                    .map(|y| y.to_string())
            })
        };
        try_seg(k + 1).or_else(|| if k > 0 { try_seg(k - 1) } else { None })
    };

    let month_year_back: Option<String> = {
        let segs_before = &segments[..idx];
        let n = segs_before.len();
        let mut found = None;
        for k in (0..n).rev() {
            if let Some(my) = month_year_from_seg(segs_before[k]) {
                if my.chars().any(|c| c.is_ascii_digit()) {
                    found = Some(my);
                } else if let Some(yr) = find_year_near(segs_before, k) {
                    found = Some(format!("{} {}", my, yr));
                } else {
                    found = Some(my);
                }
                break;
            }
        }
        found
    };

    let month_year_fwd: Option<String> = if month_year_back.is_none() {
        let segs_after = &segments[idx + 1..];
        let n = segs_after.len();
        let mut found = None;
        for k in 0..n {
            if let Some(my) = month_year_from_seg(segs_after[k]) {
                if my.chars().any(|c| c.is_ascii_digit()) {
                    found = Some(my);
                } else if let Some(yr) = find_year_near(segs_after, k) {
                    found = Some(format!("{} {}", my, yr));
                } else {
                    found = Some(my);
                }
                break;
            }
        }
        found
    } else {
        None
    };

    // If only a forward month was found, the event precedes that header and belongs
    // to the previous month (e.g. "29 GOOD FRIDAY … APRIL 2024" → March 2024).
    let month_year = match (month_year_back, month_year_fwd) {
        (Some(back), _)     => Some(back),
        (None, Some(fwd))   => prev_month_year(&fwd).or(Some(fwd)),
        (None, None)        => None,
    };

    // Reject chunks whose month+year doesn't match the question year.
    if let Some(q_year) = extract_year(question) {
        if let Some(ref my) = month_year {
            if !my.contains(q_year) {
                return None;
            }
        }
    }

    let event = event_label(segments[idx]);

    let answer = match (day, month_year) {
        (Some(d), Some(my)) => {
            if event.is_empty() {
                format!("{} {}", d, my)
            } else {
                format!("{} {} — {}", d, my, event)
            }
        }
        (Some(d), None) => {
            if event.is_empty() {
                d.clone()
            } else {
                format!("{} — {}", d, event)
            }
        }
        (None, Some(my)) => {
            if event.is_empty() {
                my.clone()
            } else {
                format!("{} ({})", event, my)
            }
        }
        (None, None) => event.clone(),
    };

    if answer.is_empty() { None } else { Some((answer, combined_score)) }
}

fn prev_month_year(month_year: &str) -> Option<String> {
    const MONTHS: [&str; 12] = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ];
    let (idx, _) = MONTHS.iter().enumerate()
        .find(|(_, m)| month_year.contains(*m))?;
    let yr: u32 = month_year.split_whitespace()
        .find(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))?
        .parse().ok()?;
    if idx == 0 {
        Some(format!("December {}", yr - 1))
    } else {
        Some(format!("{} {}", MONTHS[idx - 1], yr))
    }
}

fn parse_day(w: &str) -> Option<String> {
    if w.len() <= 2 && w.chars().all(|c| c.is_ascii_digit()) {
        w.parse::<u32>().ok()
            .filter(|&n| (1..=31).contains(&n))
            .map(|_| w.to_string())
    } else {
        None
    }
}

fn event_label(segment: &str) -> String {
    const DAYS_OF_WEEK: [&str; 7] = [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    ];
    let trimmed = segment.trim();
    let upper   = trimmed.to_uppercase();

    let after_dow = DAYS_OF_WEEK.iter()
        .find_map(|day| {
            if upper.starts_with(day) {
                Some(trimmed[day.len()..].trim_start())
            } else {
                None
            }
        })
        .unwrap_or(trimmed);

    let base = clean_segment(after_dow);
    let words: Vec<&str> = base.split_whitespace().collect();
    // Walk backward: drop trailing standalone day numbers
    let mut end = words.len();
    while end > 0 && parse_day(words[end - 1]).is_some() {
        end -= 1;
    }
    let label = words[..end].join(" ");
    dedup_trailing_phrase(&label)
}

/// Remove a repeated trailing phrase. e.g. "FOO BAR FOO BAR" → "FOO BAR"
fn dedup_trailing_phrase(label: &str) -> String {
    let words: Vec<&str> = label.split_whitespace().collect();
    let n = words.len();
    for half in 1..=(n / 2) {
        if words[n - half..] == words[n - 2 * half..n - half] {
            return words[..n - half].join(" ");
        }
    }
    label.to_string()
}

fn clean_segment(segment: &str) -> String {
    let trimmed = segment.trim();
    if let Some((first, rest)) = trimmed.split_once(' ') {
        if first.len() <= 2 && first.chars().all(|c| c.is_ascii_digit()) {
            return rest.trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Whole-word match — prevents short terms like "how" matching inside "showcase".
fn contains_word(text: &str, word: &str) -> bool {
    let tb = text.as_bytes();
    let wb = word.as_bytes();
    let wl = wb.len();
    if wl > tb.len() { return false; }
    for i in 0..=(tb.len() - wl) {
        if &tb[i..i + wl] == wb {
            let before_ok = i == 0 || !tb[i - 1].is_ascii_alphanumeric();
            let after_ok  = i + wl == tb.len() || !tb[i + wl].is_ascii_alphanumeric();
            if before_ok && after_ok { return true; }
        }
    }
    false
}
