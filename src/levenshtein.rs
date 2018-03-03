/// Simple recursive implementation of Levenshtein string edit distance.
pub fn edit_distance(s: &[char], ls: usize, t: &[char], lt: usize) -> usize {
    if ls == 0 {
        return lt;
    }

    if lt == 0 {
        return ls;
    }

    if s[ls] == t[ls] {
        return edit_distance(s, ls - 1, t, lt - 1);
    }

    let mut a = edit_distance(s, ls - 1, t, lt - 1);
    let b = edit_distance(s, ls, t, lt - 1);
    let c = edit_distance(s, ls - 1, t, lt);

    if a > b {
        a = b;
    } else if a > c {
        a = c;
    }

    a + 1
}
