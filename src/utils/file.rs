use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub fn find_images(path: &str) -> Vec<String> {
    let mut images = Vec::new();

    for entry in WalkDir::new(path) {
        if let Ok(entry) = entry {
            if let Some(extension) = entry.path().extension() {
                if let Some("jpg") | Some("jpeg") | Some("png") | Some("gif") = extension.to_str() {
                    images.push(entry.path().to_str().unwrap().to_owned());
                }
            }
        }
    }

    images
}