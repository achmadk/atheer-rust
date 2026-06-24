use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

pub struct L3CompressedStorage {
    storage_dir: PathBuf,
    compression_enabled: bool,
}

impl L3CompressedStorage {
    pub fn new(storage_dir: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&storage_dir)?;
        Ok(Self {
            storage_dir,
            compression_enabled: true,
        })
    }

    pub fn snapshot(&mut self, model_id: &str, data: &[u8]) -> std::io::Result<String> {
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let path = self
            .storage_dir
            .join(format!("{}_{}.snap", model_id, snapshot_id));

        let compressed = if self.compression_enabled {
            lz4::block::compress(data, None, true).unwrap_or_else(|_| data.to_vec())
        } else {
            data.to_vec()
        };

        let mut file = File::create(path)?;
        file.write_all(&compressed)?;

        Ok(snapshot_id)
    }

    pub fn restore(&self, snapshot_id: &str) -> std::io::Result<Vec<u8>> {
        let path = self.find_snapshot(snapshot_id)?;
        let mut file = File::open(path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        if self.compression_enabled {
            Ok(lz4::block::decompress(&compressed, None).unwrap_or(compressed))
        } else {
            Ok(compressed)
        }
    }

    fn find_snapshot(&self, snapshot_id: &str) -> std::io::Result<PathBuf> {
        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains(snapshot_id) {
                return Ok(entry.path());
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Snapshot not found: {}", snapshot_id),
        ))
    }

    pub fn size_bytes(&self) -> usize {
        fs::read_dir(&self.storage_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len() as usize)
                    .sum()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_snapshot_restore() {
        let temp_dir = env::temp_dir().join("aether_test_l3");
        let mut storage = L3CompressedStorage::new(temp_dir.clone()).unwrap();

        let data = b"test data for compression";
        let snapshot_id = storage.snapshot("test-model", data).unwrap();

        let restored = storage.restore(&snapshot_id).unwrap();
        assert_eq!(restored, data);

        fs::remove_dir_all(temp_dir).ok();
    }
}
