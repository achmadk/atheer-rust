//! CLI tool for encrypting GGUF and .mlpackage model files.
//!
//! Usage:
//!   atheer-encrypt --input model.gguf --output model.gguf.enc --key-id mykey
//!   atheer-encrypt --input model.mlpackage --output dir/ --key-id mykey --mlpackage
//!   atheer-encrypt --input model.gguf --output model.gguf.enc --key-id mykey --server-mode

use aes_gcm::{
    aead::generic_array::GenericArray,
    aead::{Aead, OsRng},
    AeadCore, Aes256Gcm, KeyInit,
};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "atheer-encrypt",
    about = "Encrypt GGUF and .mlpackage model files"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    key_id: String,

    #[arg(long)]
    mlpackage: bool,

    #[arg(long)]
    server_mode: bool,
}

fn main() {
    let args = Args::parse();

    let key = Aes256Gcm::generate_key(OsRng);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    if args.mlpackage {
        encrypt_mlpackage(&args.input, &args.output, key.as_slice(), nonce.as_slice());
    } else {
        encrypt_gguf(&args.input, &args.output, key.as_slice(), nonce.as_slice());
    }

    if args.server_mode {
        let output = serde_json::json!({
            "key_id": args.key_id,
            "key": hex::encode(key),
            "nonce": hex::encode(nonce),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        eprintln!(
            "Encrypted: {} -> {}",
            args.input.display(),
            args.output.display()
        );
        eprintln!("Key ID: {}", args.key_id);
        eprintln!(
            "Store this key in the platform Keychain/Keystore under '{}'.",
            args.key_id
        );
        eprintln!("(Use --server-mode to output the key material as JSON.)");
    }
}

fn encrypt_gguf(input: &Path, output: &Path, key: &[u8], nonce: &[u8]) {
    let plaintext = fs::read(input).expect("failed to read input");
    let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
    let nonce_arr = GenericArray::<u8, typenum::U12>::from_slice(nonce);
    let ciphertext = cipher
        .encrypt(
            nonce_arr,
            aes_gcm::aead::Payload {
                msg: &plaintext,
                aad: b"atheer-model-v1",
            },
        )
        .expect("encryption failed");

    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(nonce);
    out.extend_from_slice(&ciphertext);
    fs::write(output, &out).expect("failed to write output");
}

fn encrypt_mlpackage(input: &Path, output: &Path, key: &[u8], _nonce: &[u8]) {
    copy_dir_recursive(input, output).expect("failed to copy .mlpackage");
    encrypt_bin_files(output, key).expect("failed to encrypt .bin files");
}

fn encrypt_bin_files(dir: &Path, key: &[u8]) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                encrypt_bin_files(&path, key)?;
            } else if path.extension().map(|e| e == "bin").unwrap_or(false) {
                let plaintext = fs::read(&path)?;
                let cipher = Aes256Gcm::new_from_slice(key).expect("valid key");
                let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
                let ciphertext = cipher
                    .encrypt(
                        &nonce,
                        aes_gcm::aead::Payload {
                            msg: &plaintext,
                            aad: b"atheer-mlpackage-weight",
                        },
                    )
                    .expect("encryption failed");
                let mut out = Vec::with_capacity(12 + ciphertext.len());
                out.extend_from_slice(&nonce);
                out.extend_from_slice(&ciphertext);
                fs::write(&path, &out)?;
            }
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            copy_dir_recursive(&src_path, &dst_path)?;
        }
    } else {
        fs::copy(src, dst)?;
    }
    Ok(())
}
