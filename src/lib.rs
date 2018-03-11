#![crate_name = "fiveo"]
#![crate_type = "lib"]
// Include the unstable testing library.
#![feature(test)]
// Don't build with the standard library when targeting WebAssembly.
#![cfg_attr(feature = "webassembly", no_std)]
// We want to disable the default allocator and rely on `wee_alloc` for allocations in WebAssembly builds.  We include the
// `alloc` crate for some extra core functionality.
#![cfg_attr(feature = "webassembly", feature(alloc, core_intrinsics, core_float))]
#[cfg(feature = "webassembly")]
extern crate alloc;
#[cfg(feature = "webassembly")]
use {alloc::{BinaryHeap, Vec}, core::{cmp, iter, str, f32, num::Float}};

// When we're not targetting webassembly, import the same types from libstd.
#[cfg(not(feature = "webassembly"))]
use std::{cmp, iter, str, collections::BinaryHeap, f32};

/// A 32-bit bitmask that maps the alphabet to the first 25 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateBitmask(u32);

impl CandidateBitmask {
    pub fn from(values: &mut Iterator<Item = char>) -> Self {
        let mut bitmask = 0 as u32;

        loop {
            match values.next() {
                Some(val) if val.is_ascii() && val.is_alphabetic() => {
                    bitmask |= 1 << (val as u8 - 'a' as u8);
                }
                Some(_) => continue,
                None => break,
            }
        }

        CandidateBitmask(bitmask)
    }

    fn matches(&self, other: &Self) -> bool {
        (self.0 & other.0) == self.0
    }
}

/// A match candidate in a `Matcher`s dictionary.
#[derive(Clone, Debug, PartialEq)]
struct Candidate<'a> {
    /// The original value of this entry in the dictionary.
    value: &'a str,

    /// A bitmask used to perform quick `String::contains` operations for single characters.
    mask: CandidateBitmask,
}

impl<'a> Candidate<'a> {
    /// Create a new `Candidate` from the given string, creating a copy of it, normalizing to lowercase
    /// and generating a `CandidateBitmask`.
    fn from(value: &'a str) -> Candidate<'a> {
        let mask = CandidateBitmask::from(&mut value.to_ascii_lowercase().chars());

        Candidate { value, mask }
    }
}

/// An approximate search result from a `Search` created by a `Matcher`.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SearchResult {
    /// A reference to the `Candidate` value this result came from.
    index: usize,
    /// A score of this result ranked against the query string.
    score: f32,
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

impl SearchResult {
    /// Get a reference to the string value of this result.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the score of this result.
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// A set of parameters that can be tuned to change the scores assigned to certain matches.
pub struct MatcherParameters {
    /// The bonus for a matching character found after a slash.
    slash_bonus: f32,

    /// The bonus for a matching character found after a separator.
    separator_bonus: f32,

    /// The bonus for a matching character found as a separator of a CamelCase string.
    camelcase_bonus: f32,

    /// The bonus for a matching character found after a period.
    period_bonus: f32,

    /// The maximum gap between a search character and a match to be considered before moving to the next character.
    max_gap: usize,

    /// The size of the cache that will be allocated each search for memoizing score function calls.  Query/candidate pairs exceeding
    /// this size will only be matched using the simple matcher.
    cache_size: usize,

    /// The base distance penalty for matches occurring after proceeding without a successful match.
    distance_penalty: f32,

    /// The lowest value the distance penalty can decrease to.
    min_distance_penalty: f32,

    /// The increment that the distance penalty decreases in.
    cumulative_distance_penalty: f32,
}

/// Define a sane set of default `MatcherParameters` that adhere to the same parameters followed by Cmd-T.
///
/// The rules in these default parameters prefers slashes over separators, and camelcase / separators over periods, with a max gap of 10.
impl Default for MatcherParameters {
    fn default() -> Self {
        MatcherParameters {
            camelcase_bonus: 0.8f32,
            separator_bonus: 0.8f32,
            slash_bonus: 0.9f32,
            period_bonus: 0.7f32,
            max_gap: 10,
            cache_size: 2000,
            distance_penalty: 0.6f32,
            min_distance_penalty: 0.2f32 + f32::EPSILON,
            cumulative_distance_penalty: 0.05f32,
        }
    }
}

/// A fuzzy-search algorithm that uses bitmasks to perform fast string comparisons.
///
/// # Example:
///
/// ```rust
/// let searcher = fiveo::Matcher::new("my_word1\nmy_word2\n", fiveo::MatcherParameters::default()).unwrap();
/// // Search for "my_word1" and return a maximum of 10 results.
/// let matches = searcher.search("my_word1", 10);
/// ```
pub struct Matcher<'a> {
    /// A list of entries in this `Matcher`s dictionary.
    candidates: Vec<Candidate<'a>>,

    /// The parameters used to tune the scoring function.
    parameters: MatcherParameters,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatcherError {
    TextEncoding(str::Utf8Error),
}

impl<'a> Matcher<'a> {
    /// Create a new `Matcher` from the given input data.  The input data should contain
    /// a list of input entries delimited by newlines that a matching dictionary can be built from.
    pub fn new(dictionary: &str, parameters: MatcherParameters) -> Result<Matcher, MatcherError> {
        let candidates = dictionary
            .lines()
            .map(|line| Candidate::from(line))
            .collect();

        Ok(Matcher {
            candidates,
            parameters,
        })
    }

    /// Search for `Candidate`s that approximately match the given `query` string and return a
    /// list of `SearchResult`s.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
        let query_lowercase = query.to_lowercase();
        let query_mask = CandidateBitmask::from(&mut query_lowercase.chars());
        let mut result_heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(max_results);
        let mut match_idx_cache = Vec::with_capacity(self.parameters.cache_size);
        let mut match_score_cache = Vec::with_capacity(self.parameters.cache_size);

        for (candidate_index, candidate) in self.candidates.iter().enumerate() {
            if !query_mask.matches(&candidate.mask) {
                continue;
            }

            match_idx_cache.resize(self.parameters.cache_size, None);
            match_score_cache.resize(self.parameters.cache_size, None);

            let score = self.score_candidate(
                query,
                candidate,
                &mut match_idx_cache,
                &mut match_score_cache,
            );

            if score > 0.0f32 {
                let is_higher_scoring = result_heap
                    .peek()
                    .map(|rs| score > rs.score())
                    .unwrap_or(false);

                if is_higher_scoring || result_heap.len() < max_results {
                    result_heap.push(SearchResult {
                        index: candidate_index,
                        score,
                    })
                }

                if result_heap.capacity() > max_results {
                    result_heap.pop();
                }
            }
        }

        result_heap.into_sorted_vec()
    }

    /// Score the given `candidate` against the search query`.  Use the `match_idx_cache_` and `match_score_cache` to memoize calls
    /// to the `score_candidate_recursive` fn.
    fn score_candidate(
        &self,
        query: &str,
        candidate: &Candidate,
        match_idx_cache: &mut Vec<Option<usize>>,
        match_score_cache: &mut Vec<Option<f32>>,
    ) -> f32 {
        let query_len = query.len();
        let candidate_len = candidate.value.len();
        let score = query_len as f32
            * self.score_candidate_recursive(
                &mut query.char_indices().peekable(),
                query_len,
                &mut candidate.value.char_indices().peekable(),
                candidate_len,
                match_idx_cache,
                match_score_cache,
            );

        score.max(0.)
    }

    fn score_candidate_recursive(
        &self,
        query_chars: &mut iter::Peekable<str::CharIndices>,
        query_len: usize,
        candidate_chars: &mut iter::Peekable<str::CharIndices>,
        candidate_len: usize,
        match_idx_cache: &mut Vec<Option<usize>>,
        match_score_cache: &mut Vec<Option<f32>>,
    ) -> f32 {
        let mut score = 0.0f32;

        // Return a score of 0 if there are no characters remaining to compare.
        let (query_char_index, query_char) = match query_chars.next() {
            Some((idx, ch)) => (idx, ch),
            None => return 1.0f32,
        };

        // Peek the next character so we can calculate the distance bonus and store an entry in the matches/memo caches
        // for this call.
        let candidate_start_index = match candidate_chars.peek() {
            Some(&(idx, _)) => idx,
            None => return 1.0f32,
        };

        // Calculate the position in the memo/best match caches.
        let cache_offset = query_char_index * candidate_len + candidate_start_index;

        if let Some(cached_score) = match_score_cache[cache_offset] {
            return cached_score;
        }

        // The position of the best match.
        let mut best_match_index: Option<usize> = None;

        // The remaining number of characters to process before the gap is considered too large.
        let mut remaining_candidate_chars = self.parameters.max_gap;

        // The last character that was checked in the candidate search string.
        let mut last_candidate_char: Option<char> = None;

        // Position of the last back/forwards slash that was found.
        let mut last_slash: Option<usize> = None;

        // The growing distance penalty.
        let mut distance_penalty = self.parameters.distance_penalty;

        while let Some((candidate_char_index, candidate_char)) = candidate_chars.next() {
            if remaining_candidate_chars == 0 {
                break;
            }

            if query_char_index == 0 && (query_char == '\\' || query_char == '/') {
                last_slash = Some(candidate_char_index);
            }

            if query_char == candidate_char
                || query_char.to_lowercase().eq(candidate_char.to_lowercase())
            {
                let query_char_score = if candidate_char_index == candidate_start_index {
                    1.0f32
                } else {
                    match last_candidate_char {
                        Some(val) => match val {
                            '/' => self.parameters.slash_bonus,
                            '-' | '_' | ' ' | '0'...'9' => self.parameters.separator_bonus,
                            'a'...'z' if candidate_char.is_uppercase() => {
                                self.parameters.camelcase_bonus
                            }
                            '.' => self.parameters.period_bonus,
                            _ => distance_penalty,
                        },
                        _ => distance_penalty,
                    }
                };

                if query_char_index > 0 && distance_penalty > self.parameters.min_distance_penalty {
                    distance_penalty -= self.parameters.cumulative_distance_penalty;
                }

                let mut new_score = query_char_score
                    * self.score_candidate_recursive(
                        query_chars,
                        query_len,
                        candidate_chars,
                        candidate_len,
                        match_idx_cache,
                        match_score_cache,
                    );

                if query_char_index == 0 {
                    new_score /= (candidate_len - last_slash.unwrap_or(0)) as f32;
                }

                if new_score > score {
                    score = new_score;
                    best_match_index = Some(candidate_char_index);

                    if f32::abs(score - 1.0f32) < f32::EPSILON {
                        break;
                    }
                }
            }

            last_candidate_char = Some(candidate_char);
            remaining_candidate_chars -= 1;
        }

        // Store the results in the cache so we don't have to run this computation again during this search.
        match_score_cache[cache_offset] = Some(score);
        match_idx_cache[cache_offset] = best_match_index;

        score
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use self::test::{black_box, Bencher};

    /// Exact matches should always return a perfect score of 1.0f32 and take precedence over any other matches.
    #[test]
    pub fn exact_match_has_perfect_score() {
        let searcher = Matcher::new("my_word1", MatcherParameters::default()).unwrap();
        let results = searcher.search("my_word1", 1);

        assert_eq!(1.0f32, results[0].score());
    }

    /// A benchmark that searches through a list of 13k files from the Unreal Engine 4 source code.
    #[bench]
    fn unreal_engine_search(b: &mut Bencher) {
        let searcher = Matcher::new(
            include_str!("../benchmark_data/ue4_file_list.txt"),
            MatcherParameters::default(),
        ).unwrap();

        b.iter(|| black_box(searcher.search("file.cpp", 100)))
    }
}
