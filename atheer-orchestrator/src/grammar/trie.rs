use std::collections::HashMap;
use atheer_core::Tokenizer;

#[derive(Debug, Clone)]
pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
        }
    }

    fn insert(&mut self, text: &str) {
        let mut node = self;
        for ch in text.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.is_end = true;
    }

    fn get_all_descendants(&self, prefix: &str) -> Vec<String> {
        let mut results = Vec::new();
        let mut node = self;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(next) => node = next,
                None => return results,
            }
        }
        self.collect_all_strings(node, prefix, &mut results);
        results
    }

    fn collect_all_strings(&self, node: &TrieNode, prefix: &str, results: &mut Vec<String>) {
        if node.is_end {
            results.push(prefix.to_string());
        }
        for (ch, child) in &node.children {
            let new_prefix = format!("{}{}", prefix, ch);
            self.collect_all_strings(child, &new_prefix, results);
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarTrie {
    root: TrieNode,
    token_ids: Vec<u32>,
    token_strings: Vec<String>,
    string_to_ids: HashMap<String, Vec<u32>>,
}

impl GrammarTrie {
    pub fn build(tokenizer: &Tokenizer) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut root = TrieNode::new();
        let mut token_ids = Vec::with_capacity(vocab_size);
        let mut token_strings = Vec::with_capacity(vocab_size);
        let mut string_to_ids: HashMap<String, Vec<u32>> = HashMap::new();

        for token_id in 0..vocab_size {
            let token_text = tokenizer.decode(&[token_id as u32], false);
            root.insert(&token_text);
            token_ids.push(token_id as u32);
            token_strings.push(token_text.clone());
            string_to_ids.entry(token_text).or_default().push(token_id as u32);
        }

        Self {
            root,
            token_ids,
            token_strings,
            string_to_ids,
        }
    }

    pub fn valid_tokens(&self, prefix: &str) -> Vec<u32> {
        let matching_strings = self.root.get_all_descendants(prefix);
        let mut result = Vec::new();
        for s in matching_strings {
            if let Some(ids) = self.string_to_ids.get(&s) {
                result.extend(ids.iter().copied());
            }
        }
        result
    }

    pub fn memory_bytes(&self) -> usize {
        let string_size: usize = self.token_strings.iter().map(|s| s.capacity()).sum();
        std::mem::size_of::<Self>()
            + self.token_ids.capacity() * std::mem::size_of::<u32>()
            + string_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_memory_bounds() {
        let trie = GrammarTrie {
            root: TrieNode::new(),
            token_ids: vec![],
            token_strings: vec![],
            string_to_ids: HashMap::new(),
        };
        assert!(trie.memory_bytes() < 5_242_880);
    }
}
