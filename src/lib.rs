#![crate_name = "fiveo"]
#![crate_type = "lib"]

use std::str;
use std::iter;

trait CandidateBitmaskOffset: Sized {
    fn offset(&self) -> Result<usize, &'static str>;
}

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
#[derive(Debug)]
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
}

impl PartialEq for CandidateBitmask {
    fn eq(&self, other: &CandidateBitmask) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl Eq for CandidateBitmask {}

#[derive(Debug, PartialEq, Eq)]
struct Candidate {
    value: String,
    lowercase: String,
    mask: CandidateBitmask,
}

impl Candidate {
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

/// A fuzzy-search algorithm that uses bitmasks to perform fast string comparisons.
///
/// Example:
/// ```rust
/// let searcher = fuzzed::FuzzySearch::new(b"my_word1\nmy_word2\n").unwrap();
/// let first_match = searcher.search("word").next().unwrap();
///
/// assert_eq!("my_word1", first_match.value());
/// assert_eq!(0.0f32, first_match.score());
/// ```
pub struct Matcher {
    candidates: Vec<Candidate>,
}

impl Matcher {
    pub fn new(input_data: &[u8]) -> Result<Matcher, &'static str> {
        let dictionary = unsafe { str::from_utf8_unchecked(input_data) };
        let candidates = dictionary
            .lines()
            .map(|line| Candidate::from(&line))
            .collect();

        Ok(Matcher { candidates })
    }

    pub fn search<'a>(&'a self, query: &'a str) -> Search<'a> {
        Search::new(query, &self.candidates)
    }
}

pub struct Search<'a> {
    query: &'a str,
    query_bitmask: CandidateBitmask,
    candidates: &'a [Candidate],
    pos: usize,
}

impl<'a> Search<'a> {
    fn new(query: &'a str, candidates: &'a [Candidate]) -> Search<'a> {
        Search {
            query,
            query_bitmask: CandidateBitmask::from(&mut query.chars()),
            candidates,
            pos: 0,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SearchResult<'a> {
    value: &'a str,
    score: f32,
}

impl<'a> SearchResult<'a> {
    pub fn value(&'a self) -> &'a str {
        self.value
    }

    pub fn score(&self) -> f32 {
        self.score
    }
}

impl<'a> Iterator for Search<'a> {
    type Item = SearchResult<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pos > self.candidates.len() -1 {
                return None;
            }

            let candidate = &self.candidates[self.pos];
            self.pos = self.pos + 1;

            if candidate.mask == self.query_bitmask {
                return Some(SearchResult {
                    value: candidate.value.as_str(),
                    score: 0.0f32,
                });
            }
        }
    }
}
