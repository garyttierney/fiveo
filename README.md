# fiveo

[![LICENSE](https://img.shields.io/badge/license-ISC-blue.svg)](LICENSE)
[![Build Status](https://travis-ci.org/garyttierney/fiveo.svg?branch=master)](https://travis-ci.org/garyttierney/fiveo)
[![Crates.io Version](https://img.shields.io/crates/v/fiveo.svg)](https://crates.io/crates/fiveo)

`fiveo` is a fuzzy text searching library built for use on the web.  The API is designed with usage from a C-like foreign function interface in mind so it can be easily embedded in WebAssembly objects.

## Example

```rust
extern crate fiveo;

use fiveo::Matcher;
use fiveo::MatcherParameters;

fn main() {
    // Create a new matcher with a single entry.
    let searcher = fiveo::Matcher::new("/this/is/a/test/dir\n", MatcherParameters::default()).unwrap();
    // Search for "tiatd" and return a maximum of 10 results.
    let matches = searcher.search("tiatd", 10);

    assert_eq!(0, matches[0].index());
    assert_eq!(1.0f32, matches[0].score());
}
```

## Documentation

- [Reference documentation](https://docs.rs/fiveo).

## Installation

`fiveo` can be installed using Cargo via the crate available on [crates.io](https://crates.io/fiveo)

```toml
[dependencies]
five = "0.2.0"
```

By default `fiveo` will be built and linked against the Rust standard library.  For usage in WebAssembly builds there is a compilation feature
available to switch to dependencies on `liballoc` and `libcore`.

You can activate those features like this:

```toml
[dependencies.fiveo]
version = "0.2.0"
features = ["webassembly"]
```

## Credits

`fiveo` is inspired by the Sublime fuzzy text matcher and [@hansonw](https://github.com/hansonw/fuzzy-native)'s port of Cmd-T's algorithm