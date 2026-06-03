/// Parses the internal heap info string from HPROF STRING record with id=0.
/// Extracts JVM oop size, object alignment, and GC generation address ranges.
/// Equivalent to Java's HprofHeader.
#[derive(Debug, Clone)]
pub struct HeapInfo {
    pub oop_size: u8,
    pub object_alignment: u8,
    pub new_gen_start: u64,
    pub new_gen_end: u64,
    pub eden_start: u64,
    pub eden_end: u64,
    pub from_start: u64,
    pub from_end: u64,
    pub to_start: u64,
    pub to_end: u64,
    pub old_gen_start: u64,
    pub old_gen_end: u64,
    pub perm_gen_start: u64,
    pub perm_gen_end: u64,
}

impl HeapInfo {
    /// Parse the heap info string. Format example:
    /// "1.2 heapOopSize 8 MinObjAlignmentInBytes 8 ParNew [0x00000000b0000000-0x00000000b8000000) ..."
    pub fn parse(input: &str) -> Option<Self> {
        let mut info = Self {
            oop_size: 0,
            object_alignment: 8,
            new_gen_start: 0, new_gen_end: 0,
            eden_start: 0, eden_end: 0,
            from_start: 0, from_end: 0,
            to_start: 0, to_end: 0,
            old_gen_start: 0, old_gen_end: 0,
            perm_gen_start: 0, perm_gen_end: 0,
        };

        let tokens: Vec<&str> = Tokenizer::new(input).collect();
        if tokens.is_empty() || tokens[0] != "1.2" {
            return None;
        }

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "heapOopSize" => {
                    i += 1;
                    if i < tokens.len() {
                        info.oop_size = tokens[i].parse().unwrap_or(0);
                    }
                }
                "MinObjAlignmentInBytes" => {
                    i += 1;
                    if i < tokens.len() {
                        info.object_alignment = tokens[i].parse().unwrap_or(8);
                    }
                }
                "ParNew" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.new_gen_start = s;
                        info.new_gen_end = e;
                    }
                }
                "CMS" | "OLD" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.old_gen_start = s;
                        info.old_gen_end = e;
                    }
                }
                "PERM" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.perm_gen_start = s;
                        info.perm_gen_end = e;
                    }
                }
                "EDEN" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.eden_start = s;
                        info.eden_end = e;
                    }
                }
                "FROM" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.from_start = s;
                        info.from_end = e;
                    }
                }
                "TO" => {
                    if let Some((s, e)) = parse_range(&tokens, &mut i) {
                        info.to_start = s;
                        info.to_end = e;
                    }
                }
                // Skip tokens we don't care about
                _ => {}
            }
            i += 1;
        }

        if info.oop_size < 1 || info.oop_size > 8 || info.object_alignment == 0 {
            return None;
        }

        Some(info)
    }

    /// Classify an object address into a GC generation.
    /// Returns 1 for young gen, 2 for old/perm gen, 0 for unknown.
    pub fn get_generation(&self, address: u64) -> u8 {
        if self.in_range(address, self.new_gen_start, self.new_gen_end)
            || self.in_range(address, self.eden_start, self.eden_end)
            || self.in_range(address, self.from_start, self.from_end)
            || self.in_range(address, self.to_start, self.to_end)
        {
            1
        } else if self.old_gen_start > 0 || self.perm_gen_start > 0 {
            2
        } else {
            0
        }
    }

    fn in_range(&self, addr: u64, start: u64, end: u64) -> bool {
        start > 0 && addr >= start && addr < end
    }
}

/// Tokenizer that splits on whitespace, `[`, `]`, `-`.
struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() {
            let c = bytes[self.pos];
            if c == b' ' || c == b'[' || c == b']' || c == b'-' {
                self.pos += 1;
                continue;
            }
            let start = self.pos;
            while self.pos < bytes.len() {
                let c = bytes[self.pos];
                if c == b' ' || c == b'[' || c == b']' || c == b'-' {
                    break;
                }
                self.pos += 1;
            }
            if start < self.pos {
                return Some(&self.input[start..self.pos]);
            }
        }
        None
    }
}

fn parse_range(tokens: &[&str], i: &mut usize) -> Option<(u64, u64)> {
    *i += 1;
    if *i >= tokens.len() {
        return None;
    }
    let start = parse_hex(tokens[*i])?;
    *i += 1;
    if *i >= tokens.len() {
        return None;
    }
    let end = parse_hex(tokens[*i])?;
    Some((start, end))
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_heap_info() {
        let input = "1.2 heapOopSize 8 MinObjAlignmentInBytes 8 ParNew 0xb0000000 0xb8000000 EDEN 0xb0000000 0xb4000000 FROM 0xb4000000 0xb6000000 TO 0xb6000000 0xb8000000 CMS 0xc0000000 0x100000000";
        let info = HeapInfo::parse(input).unwrap();
        assert_eq!(info.oop_size, 8);
        assert_eq!(info.object_alignment, 8);
        assert_eq!(info.eden_start, 0xb0000000);
    }

    #[test]
    fn test_get_generation() {
        let info = HeapInfo {
            oop_size: 8,
            object_alignment: 8,
            eden_start: 0x1000, eden_end: 0x2000,
            old_gen_start: 0x4000, old_gen_end: 0x8000,
            new_gen_start: 0x1000, new_gen_end: 0x3000,
            from_start: 0x2000, from_end: 0x2800,
            to_start: 0x2800, to_end: 0x3000,
            perm_gen_start: 0, perm_gen_end: 0,
        };
        assert_eq!(info.get_generation(0x1500), 1); // eden
        assert_eq!(info.get_generation(0x5000), 2); // old gen
        assert_eq!(info.get_generation(0x0050), 0); // unknown
    }
}
