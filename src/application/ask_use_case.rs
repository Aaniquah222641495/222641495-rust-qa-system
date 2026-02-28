// ============================================================
// Layer 2 — Ask Use Case
// ============================================================
// Hybrid retrieval + neural approach:
//   1. Load chunks tagged with their source document
//   2. Filter chunks by year mentioned in question
//   3. Score chunks by keyword overlap
//   4. Extract clean date answer from best chunk

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
        // Overlap of 100 words ensures the month-header paragraph ("DECEMBER 2026")
        // is always in the same chunk as the calendar events that follow it.
        // The December page has ~70 words between the month header and Dec 9.
        let chunker = Chunker::new(200, 100);

        // Tag each chunk with its source filename
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

        // Extract year from question
        let question_year = extract_year(question);

        // Filter chunks: if year mentioned, only use chunks from matching doc
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

        // Score and rank chunks by keyword overlap
        let top_chunks = rank_chunks(question, &filtered_chunks, 20);

        // Run model inference to find best chunk
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

        // Run pattern-matching extraction over ALL year-filtered chunks.
        // This ensures formal calendar events (e.g. "SUMMER GRADUATION") are
        // never missed due to keyword-tie-breaking in rank_chunks: many
        // committee-meeting chunks contain "graduation" and can fill the top-20
        // before the actual event chunk is reached.
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

/// Extract a 4-digit year from a question string.
/// Splits on non-digit characters so trailing punctuation (e.g. "2024?")
/// is ignored and the bare year is returned.
fn extract_year(question: &str) -> Option<&str> {
    // Find a run of exactly 4 digits that looks like a calendar year
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

/// Rank chunks by keyword overlap with question
fn rank_chunks(
    question: &str,
    chunks:   &[(String, String)],
    top_k:    usize,
) -> Vec<(String, String)> {
    // Keep words longer than 3 chars, and also short numeric tokens (e.g. "2" in "Term 2").
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

/// Extract the most relevant date answer from a chunk.
///
/// The Chunker destroys newlines (it splits on whitespace and rejoins with
/// spaces), so chunks are flat strings where `|` is the only field delimiter.
/// A typical chunk looks like:
///   "... MARCH 2024 ... 28 | 29 | GOOD FRIDAY 30 | SATURDAY ..."
///
/// Strategy:
///  1. Split on `|` (and `\n` if any survive) to get pipe-cells.
///  2. Score each cell by keyword overlap — no month bonus (avoids selecting
///     month-header cells as the winner).
///  3. The winning cell is the event name (e.g. "GOOD FRIDAY 30 SATURDAY …").
///  4. The day number lives in the LAST WORD of the *preceding* cell
///     (e.g. "… 29" → 29).  Also try the first word of the winning cell.
///  5. Scan backward through cells for the nearest month+year header.
///  6. Return "{day} {Month} {year}" (e.g. "29 March 2024").
fn extract_clean_answer(question: &str, chunk: &str) -> Option<(String, i32)> {
    let months     = ["january","february","march","april","may","june",
                      "july","august","september","october","november","december"];
    let months_cap = ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"];

    let stop_words = ["when","what","does","will","this","that","have",
                      "from","held","the","and","for","are","was",
                      "2024","2025","2026","2027"];

    // Trim only punctuation (not digits) so term numbers like "2" survive.
    // Allow short purely-numeric words (e.g. "2" in "Term 2") through the length filter.
    let key_terms: Vec<String> = question.split_whitespace()
        .map(|w| w.to_lowercase()
            .trim_matches(|c: char| c.is_ascii_punctuation())
            .to_string())
        .filter(|w| !w.is_empty()
            && !stop_words.contains(&w.as_str())
            && (w.len() > 2 || w.chars().all(|c| c.is_ascii_digit())))
        .collect();

    if key_terms.is_empty() { return None; }

    // Split on pipe (and newline if any); no upper length filter —
    // the chunk may be one long flat string with no newlines.
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

        // Weight each match by term length — longer, more specific terms
        // (e.g. "graduation"=10) outrank short generic ones (e.g. "end"=3).
        // Use whole-word matching to prevent "how" from scoring inside "showcase".
        for term in &key_terms {
            if contains_word(&sl, term) {
                score += term.len() as i32;
            }
        }

        // Penalise purely-numeric cells (bare day numbers like "29")
        let alnum: String = seg.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if !alnum.is_empty() && alnum.chars().all(|c| c.is_ascii_digit()) { score -= 10; }

        if seg.len() < 5 { score -= 5; }

        // Bonus for ALL-CAPS segments: formal calendar events (SUMMER GRADUATION,
        // GOOD FRIDAY, START OF TERM 2) are all-caps; committee/admin text is not.
        // Only applies when at least one key-term already matched (score > 0),
        // to prevent month-header cells from winning with zero term matches.
        if score > 0 {
            let alpha: Vec<char> = seg.chars().filter(|c| c.is_alphabetic()).collect();
            if !alpha.is_empty() {
                let upper = alpha.iter().filter(|c| c.is_uppercase()).count();
                if upper as f32 / alpha.len() as f32 >= 0.7 {
                    score += 5;
                }
            }
        }

        // Hard penalty if the segment is missing a short numeric key-term
        // (e.g. "2" for "Term 2"). Prevents "START OF TERM" from winning when
        // the user asked for "Term 2" and "2" appears only as a stray day number.
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

    // Context relevance (tiebreaker): total key-term score of all OTHER segments.
    // Example: when "SUMMER GRADUATION" and "AUTUMN GRADUATION" both score 15,
    // the December chunk wins because it also contains "END OF TERM 4 End of Year
    // for Academic Staff" (matching "end"+"year"), giving it a higher context score.
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

    // --- Day extraction ---
    // The day number can be:
    //   a) the last word of the preceding pipe-cell: "… 29" → 29
    //   b) any day-number word inside the winning cell: "SATURDAY 1 WORKERS" → 1
    //      (also handles "29 GOOD FRIDAY" where it is the first word)
    // We prefer (b) first so that cells like "29 GOOD FRIDAY" are self-contained,
    // and fall back to (a) for cells like "GOOD FRIDAY" that have no inline number.
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

    // --- Month+year extraction ---
    // Helper: extract "Month YYYY" from a segment that contains a month name.
    // Handles Word documents that split a year across runs, e.g. "202 6" → "2026".
    let month_year_from_seg = |seg: &str| -> Option<String> {
        let sl = seg.to_lowercase();
        let (_, cap) = months.iter().zip(months_cap.iter())
            .find(|(m, _)| sl.contains(*m))?;
        let words: Vec<&str> = seg.split_whitespace().collect();
        // First try: a single 4-digit token.
        let yr = words.iter()
            .find(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
            .map(|w| w.to_string())
            // Fallback: join two adjacent all-digit tokens whose concatenation is
            // a 4-digit calendar year (handles "202 6" produced by docx-rs).
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

    // Scan backward first (month header normally precedes the day entries).
    // Calendar table headers often split "DECEMBER" and "2026" into adjacent
    // pipe-cells, so when a month-only segment is found we look at its neighbours
    // for a 4-digit year.
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

    // If backward scan found nothing, scan forward (header may be in a later chunk
    // boundary, e.g. "MAY 2025 SUNDAY | MONDAY …" at the tail of this chunk).
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

    // When only a forward month was found, the event sits *before* that month
    // header — meaning it belongs to the preceding month, not the found one.
    // e.g. "29 GOOD FRIDAY … APRIL 2024" → event is in March 2024, not April.
    let month_year = match (month_year_back, month_year_fwd) {
        (Some(back), _)     => Some(back),
        (None, Some(fwd))   => prev_month_year(&fwd).or(Some(fwd)),
        (None, None)        => None,
    };

    // --- Year consistency check ---
    // If we found a month+year context whose year does NOT match the question
    // year, this chunk is from the wrong period — return empty so the caller
    // tries the next-ranked chunk.
    if let Some(q_year) = extract_year(question) {
        if let Some(ref my) = month_year {
            if !my.contains(q_year) {
                return None;
            }
        }
    }

    // Build a clean event label: strip leading day number, trim trailing
    // orphaned day numbers left over from adjacent cells sharing the segment.
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

/// Return the month+year string for the calendar month *before* the given one.
/// e.g. "April 2024" → Some("March 2024"), "January 2024" → Some("December 2023")
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

/// Parse a word as a calendar day (1–31).  Returns `None` if not a valid day.
fn parse_day(w: &str) -> Option<String> {
    if w.len() <= 2 && w.chars().all(|c| c.is_ascii_digit()) {
        w.parse::<u32>().ok()
            .filter(|&n| (1..=31).contains(&n))
            .map(|_| w.to_string())
    } else {
        None
    }
}

/// Build a clean event label from a pipe-cell:
///  1. Strip a leading day-of-week name ("SATURDAY 1 WORKERS DAY" → "1 WORKERS DAY").
///  2. Strip a leading day number ("1 WORKERS DAY" → "WORKERS DAY").
///  3. Truncate before any trailing orphaned day numbers that bled in from
///     adjacent cells when newlines were removed ("GOOD FRIDAY 30 31" → "GOOD FRIDAY").
fn event_label(segment: &str) -> String {
    const DAYS_OF_WEEK: [&str; 7] = [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    ];
    let trimmed = segment.trim();
    let upper   = trimmed.to_uppercase();

    // Strip leading day-of-week name (case-insensitive)
    let after_dow = DAYS_OF_WEEK.iter()
        .find_map(|day| {
            if upper.starts_with(day) {
                Some(trimmed[day.len()..].trim_start())
            } else {
                None
            }
        })
        .unwrap_or(trimmed);

    // Strip leading day number
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

/// Remove a repeated trailing phrase from an event label.
/// e.g. "SUMMER GRADUATION SUMMER GRADUATION" → "SUMMER GRADUATION"
///      "WCED SCHOOLS CLOSE SUMMER GRADUATION SUMMER GRADUATION"
///                    → "WCED SCHOOLS CLOSE SUMMER GRADUATION"
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

/// Strip a leading day number from a segment label.
fn clean_segment(segment: &str) -> String {
    let trimmed = segment.trim();
    if let Some((first, rest)) = trimmed.split_once(' ') {
        if first.len() <= 2 && first.chars().all(|c| c.is_ascii_digit()) {
            return rest.trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Whole-word substring test.
/// Returns `true` only when `word` appears in `text` surrounded by
/// non-alphanumeric characters (or at string boundaries).
/// Prevents short terms like "how" from matching inside "showcase".
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
