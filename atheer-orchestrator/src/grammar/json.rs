use super::GrammarConstraint;

/// Tracks structural nesting: objects and arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

/// Position inside a JSON value production.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Pos {
    /// Expecting any JSON value (root, array element, object value after `:`)
    Value,
    /// After `{`, expecting first member key string (or `}` for empty)
    ObjKey,
    /// After `,` in object, expecting next key string
    ObjKeyComma,
    /// After key string, expecting `:`
    ObjColon,
    /// After `:` in object, expecting value
    ObjValue,
    /// After object value, expecting `,` or `}`
    ObjAfter,
    /// After `[`, expecting first element (or `]` for empty)
    ArrValue,
    /// After `,` in array, expecting next element
    ArrValueComma,
    /// After array element, expecting `,` or `]`
    ArrAfter,
    /// Inside a string literal (value position)
    String,
    /// Inside an object-key string literal
    StringKey,
    /// Inside a string after `\`
    StringEsc,
    /// Inside `\uXXXX`
    StringU(u8),
    /// Inside a number literal
    Number,
    /// Matching `true`
    True(u8),
    /// Matching `false`
    False(u8),
    /// Matching `null`
    Null(u8),
}

/// Character-level JSON grammar constraint.
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    pos: Pos,
    /// Remember the string variant when we enter an escape sequence.
    string_pos: Option<Pos>,
    stack: Vec<Container>,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    pub fn new() -> Self {
        Self {
            pos: Pos::Value,
            string_pos: None,
            stack: Vec::new(),
        }
    }

    fn pop_container(&mut self) {
        self.stack.pop();
        self.pos = match self.stack.last() {
            Some(Container::Object) => Pos::ObjAfter,
            Some(Container::Array) => Pos::ArrAfter,
            None => Pos::Value,
        };
    }

    fn enter_value(&mut self, ch: char) -> bool {
        match ch {
            '{' => {
                self.stack.push(Container::Object);
                self.pos = Pos::ObjKey;
                true
            }
            '[' => {
                self.stack.push(Container::Array);
                self.pos = Pos::ArrValue;
                true
            }
            '"' => {
                self.string_pos = Some(Pos::String);
                self.pos = Pos::String;
                true
            }
            't' => {
                self.pos = Pos::True(1);
                true
            }
            'f' => {
                self.pos = Pos::False(1);
                true
            }
            'n' => {
                self.pos = Pos::Null(1);
                true
            }
            '-' | '0'..='9' => {
                self.pos = Pos::Number;
                true
            }
            _ => false,
        }
    }

    fn feed(&mut self, ch: char) -> bool {
        match self.pos {
            // ── Value entry points ───────────────────────────────────────
            Pos::Value | Pos::ObjValue => self.enter_value(ch),

            // ── Object key positions ─────────────────────────────────────
            Pos::ObjKey => match ch {
                '"' => {
                    self.string_pos = Some(Pos::StringKey);
                    self.pos = Pos::StringKey;
                    true
                }
                '}' => {
                    self.pop_container();
                    true
                }
                _ => false,
            },
            Pos::ObjKeyComma => match ch {
                '"' => {
                    self.string_pos = Some(Pos::StringKey);
                    self.pos = Pos::StringKey;
                    true
                }
                _ => false,
            },
            Pos::ObjColon => {
                if ch == ':' {
                    self.pos = Pos::ObjValue;
                    true
                } else {
                    false
                }
            }
            Pos::ObjAfter => match ch {
                ',' => {
                    self.pos = Pos::ObjKeyComma;
                    true
                }
                '}' => {
                    self.pop_container();
                    true
                }
                _ => false,
            },

            // ── Array positions ──────────────────────────────────────────
            Pos::ArrValue => match ch {
                ']' => {
                    self.pop_container();
                    true
                }
                _ => self.enter_value(ch),
            },
            Pos::ArrValueComma => self.enter_value(ch),
            Pos::ArrAfter => match ch {
                ',' => {
                    self.pos = Pos::ArrValueComma;
                    true
                }
                ']' => {
                    self.pop_container();
                    true
                }
                _ => false,
            },

            // ── String (value) ───────────────────────────────────────────
            Pos::String => match ch {
                '"' => {
                    self.pos = match self.stack.last() {
                        Some(Container::Object) => Pos::ObjAfter,
                        Some(Container::Array) => Pos::ArrAfter,
                        None => Pos::Value,
                    };
                    true
                }
                '\\' => {
                    self.pos = Pos::StringEsc;
                    true
                }
                c => (c as u32) >= 0x20,
            },

            // ── String (object key) ──────────────────────────────────────
            Pos::StringKey => match ch {
                '"' => {
                    self.pos = Pos::ObjColon;
                    true
                }
                '\\' => {
                    self.pos = Pos::StringEsc;
                    true
                }
                c => (c as u32) >= 0x20,
            },

            // ── String escape ────────────────────────────────────────────
            Pos::StringEsc => match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                    self.pos = self.string_pos.clone().unwrap_or(Pos::String);
                    true
                }
                'u' => {
                    self.pos = Pos::StringU(0);
                    true
                }
                _ => false,
            },

            Pos::StringU(n) => {
                if ch.is_ascii_hexdigit() {
                    if n < 3 {
                        self.pos = Pos::StringU(n + 1);
                    } else {
                        self.pos = self.string_pos.clone().unwrap_or(Pos::String);
                    }
                    true
                } else {
                    false
                }
            }

            // ── Number ───────────────────────────────────────────────────
            Pos::Number => match ch {
                '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => true,
                ',' => {
                    self.pos = match self.stack.last() {
                        Some(Container::Object) => Pos::ObjKeyComma,
                        Some(Container::Array) => Pos::ArrValueComma,
                        None => return false,
                    };
                    true
                }
                '}' => {
                    if self.stack.last() == Some(&Container::Object) {
                        self.pop_container();
                        true
                    } else {
                        false
                    }
                }
                ']' if self.stack.last() == Some(&Container::Array) => {
                    self.pop_container();
                    true
                }
                _ => false,
            },

            // ── Literals ─────────────────────────────────────────────────
            Pos::True(n) => {
                let target = ['t', 'r', 'u', 'e'];
                if n < 4 && ch == target[n as usize] {
                    if n == 3 {
                        self.pos = match self.stack.last() {
                            Some(Container::Object) => Pos::ObjAfter,
                            Some(Container::Array) => Pos::ArrAfter,
                            None => Pos::Value,
                        };
                    } else {
                        self.pos = Pos::True(n + 1);
                    }
                    true
                } else {
                    false
                }
            }
            Pos::False(n) => {
                let target = ['f', 'a', 'l', 's', 'e'];
                if n < 5 && ch == target[n as usize] {
                    if n == 4 {
                        self.pos = match self.stack.last() {
                            Some(Container::Object) => Pos::ObjAfter,
                            Some(Container::Array) => Pos::ArrAfter,
                            None => Pos::Value,
                        };
                    } else {
                        self.pos = Pos::False(n + 1);
                    }
                    true
                } else {
                    false
                }
            }
            Pos::Null(n) => {
                let target = ['n', 'u', 'l', 'l'];
                if n < 4 && ch == target[n as usize] {
                    if n == 3 {
                        self.pos = match self.stack.last() {
                            Some(Container::Object) => Pos::ObjAfter,
                            Some(Container::Array) => Pos::ArrAfter,
                            None => Pos::Value,
                        };
                    } else {
                        self.pos = Pos::Null(n + 1);
                    }
                    true
                } else {
                    false
                }
            }
        }
    }
}

impl GrammarConstraint for JsonGrammar {
    fn is_valid_prefix(&self, text: &str) -> bool {
        let mut g = self.clone();
        for ch in text.chars() {
            if !g.feed(ch) {
                return false;
            }
        }
        true
    }

    fn advance(&mut self, text: &str) {
        for ch in text.chars() {
            self.feed(ch);
        }
    }

    fn reset(&mut self) {
        self.pos = Pos::Value;
        self.stack.clear();
    }

    fn clone_box(&self) -> Box<dyn GrammarConstraint> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_object() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("{}"));
    }

    #[test]
    fn test_empty_array() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("[]"));
    }

    #[test]
    fn test_object_member() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("{\"k\":1}"));
        assert!(g.is_valid_prefix("{\"a\":true,\"b\":false}"));
    }

    #[test]
    fn test_nested() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("{\"a\":[1,{\"b\":2}]}"));
    }

    #[test]
    fn test_string_escapes() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("\"hello\\n\""));
        assert!(g.is_valid_prefix("\"\\u0041\""));
    }

    #[test]
    fn test_invalid() {
        let g = JsonGrammar::new();
        assert!(!g.is_valid_prefix("{invalid}"));
        assert!(!g.is_valid_prefix("[1,]"));
        assert!(!g.is_valid_prefix("{\"k\":}"));
    }

    #[test]
    fn test_numbers() {
        let g = JsonGrammar::new();
        assert!(g.is_valid_prefix("42"));
        assert!(g.is_valid_prefix("-7.5"));
        assert!(g.is_valid_prefix("1e10"));
    }

    #[test]
    fn test_advance_and_clone() {
        let mut g = JsonGrammar::new();
        g.advance("true");
        let c = g.clone_box();
        assert!(c.is_valid_prefix(""));
    }
}
