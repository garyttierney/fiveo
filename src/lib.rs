#![crate_name = "fiveo"]
#![crate_type = "lib"]

extern crate strsim;

use std::cmp;
use std::str;
use std::iter;
use std::collections::BinaryHeap;

/// A trait that defines a type that can be mapped to a bitmap offset used
/// to create a fast `String::contains` mask for a dictionhary entry.
trait CandidateBitmaskOffset: Sized {
    fn offset(&self) -> Result<usize, &'static str>;
}

/// A bitmask offset implementation for ASCII characters that treats the 'a' - 'z' range as
/// bits 1-26.
impl CandidateBitmaskOffset for char {
    fn offset(&self) -> Result<usize, &'static str> {
        if self.is_ascii() && self.is_alphabetic() {
            Ok((*self as u8 - 'a' as u8) as usize)
        } else {
            Err("Not an ASCII character")
        }
    }
}

/// A 32-bit bitmask that maps the alphabet to the first 25 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateBitmask(u32);

impl CandidateBitmask {
    pub fn from<T>(values: &mut Iterator<Item = T>) -> Self
    where
        T: CandidateBitmaskOffset,
    {
        let mut bitmask = 0 as u32;

        loop {
            match values.next() {
                Some(val) => match val.offset() {
                    Ok(offset) => bitmask |= 1 << offset,
                    _ => continue,
                },
                None => break,
            }
        }

        CandidateBitmask(bitmask)
    }

    fn matches(&self, other: &Self) -> bool {
        (&self.0 & other.0) == self.0
    }
}

/// A match candidate in a `Matcher`s dictionary.
#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    /// The original value of this entry in the dictionary.
    value: String,

    /// The value of this entry normalized to lowercase.
    lowercase: String,

    /// A bitmask used to perform quick `String::contains` operations for single characters.
    mask: CandidateBitmask,
}

impl Candidate {
    /// Create a new `Candidate` from the given string, creating a copy of it, normalizing to lowercase
    /// and generating a `CandidateBitmask`.
    fn from(value_ref: &str) -> Candidate {
        let value = value_ref.to_owned();
        let lowercase = value_ref.to_lowercase();
        let mask = CandidateBitmask::from(&mut lowercase.chars());

        Candidate {
            value,
            lowercase,
            mask,
        }
    }
}

/// An approximate search result from a `Search` created by a `Matcher`.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SearchResult<'a> {
    /// A reference to the `Candidate` value this result came from.
    value: &'a str,
    /// A score of this result ranked against the query string.
    score: f32,
}

impl<'a> Eq for SearchResult<'a> {}

impl<'a> Ord for SearchResult<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

impl<'a> SearchResult<'a> {
    /// Get a reference to the string value of this result.
    pub fn value(&'a self) -> &'a str {
        self.value
    }

    /// Get the score of this result.
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// A fuzzy-search algorithm that uses bitmasks to perform fast string comparisons.
///
/// Example:
/// ```rust
/// let searcher = fiveo::Matcher::new("my_word1\nmy_word2\n").unwrap();
/// // Search for "my_word1" and return a maximum of 10 results.
/// let matches = searcher.search("my_word1", 10);
///
/// assert_eq!("my_word1", matches[0].value());
/// assert_eq!(1.0f32, matches[0].score());
/// assert_eq!("my_word2", matches[1].value());
/// assert!(matches[1].score() < 1.0f32);
/// ```
pub struct Matcher {
    /// A list of entries in this `Matcher`s dictionary.
    candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatcherError {
    TextEncoding(str::Utf8Error)
}

impl From<MatcherError> for u32 {
    fn from(t: MatcherError) -> u32 {
        match t {
            MatcherError::TextEncoding(_) => 1,
        }
    }
}

impl Matcher {
    /// Create a new `Matcher` from the given input data.  The input data should contain
    /// a list of input entries delimited by newlines that a matching dictionary can be built from.
    pub fn new(dictionary: &str) -> Result<Matcher, MatcherError> {
        let candidates = dictionary
            .lines()
            .map(|line| Candidate::from(&line))
            .collect();

        Ok(Matcher { candidates })
    }

    /// Search for `Candidate`s that approximately match the given `query` string and return a
    /// list of `SearchResult`s.
    pub fn search<'a>(&'a self, query: &'a str, max_results: usize) -> Vec<SearchResult<'a>> {
        let query_lowercase = query.to_lowercase();
        let query_mask = CandidateBitmask::from(&mut query_lowercase.chars());
        let mut result_heap: BinaryHeap<SearchResult<'a>> = BinaryHeap::with_capacity(max_results);

        for candidate in &self.candidates {
            if !query_mask.matches(&candidate.mask) {
                continue;
            }

            let edit_distance = strsim::levenshtein(&query_lowercase, &candidate.value);
            let longest_len = cmp::max(query.len(), candidate.value.len());
            let score = (longest_len - edit_distance) as f32 / longest_len as f32;

            if score > 0.0f32 {
                let is_higher_scoring = result_heap
                    .peek()
                    .map(|rs| score > rs.score())
                    .unwrap_or(false);

                if is_higher_scoring || result_heap.len() < max_results {
                    result_heap.push(SearchResult {
                        value: &candidate.value,
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
}
