//! OpenAI-compatible image generation adapter used by the All-in-One edition.

use crate::contracts::{ImageGenerationBackend, ImageRequest, ImageResult};
use base64::Engine;
use reqwest::blocking::{Client, Response};
use serde_json::{json, Value};
use std::io::Read;
use std::time::Duration;

const MAX_IMAGE_RESPONSE_BYTES: usize = 96 * 1024 * 1024;
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_IMAGE_ERROR_CHARS: usize = 500;

pub struct OpenAiImageBackend {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

impl OpenAiImageBackend {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into().trim().trim_end_matches('/').to_string(),
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            client: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("the HTTP client configuration is valid"),
        }
    }

    fn endpoint(&self) -> String {
        if self.base_url.ends_with("/images/generations") {
            self.base_url.clone()
        } else if self.base_url.ends_with("/v1") {
            format!("{}/images/generations", self.base_url)
        } else {
            format!("{}/v1/images/generations", self.base_url)
        }
    }
}

fn read_bounded_bytes(resp: Response, limit: usize) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    resp.take((limit as u64).saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|error| format!("Image response could not be read: {error}"))?;
    if buf.len() > limit {
        return Err(format!("Image response exceeded {limit} byte limit"));
    }
    Ok(buf)
}

pub fn detect_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) || bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

impl ImageGenerationBackend for OpenAiImageBackend {
    fn available(&self) -> bool {
        let lower = self.base_url.to_ascii_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://")
    }

    fn generate(&self, request: &ImageRequest) -> Result<ImageResult, String> {
        if !self.available() {
            return Err("Image endpoint must start with http:// or https://".into());
        }
        let body = json!({
            "model": request.model,
            "prompt": request.prompt,
            "size": request.size,
            "response_format": "b64_json",
        });
        let mut builder = self
            .client
            .post(self.endpoint())
            .timeout(Duration::from_secs(300))
            .header("Content-Type", "application/json")
            .body(body.to_string());
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        let response = builder
            .send()
            .map_err(|error| format!("Image generation failed: {error}"))?;
        let status = response.status();
        let raw_bytes = read_bounded_bytes(response, MAX_IMAGE_RESPONSE_BYTES)?;
        let payload = String::from_utf8_lossy(&raw_bytes);
        if !status.is_success() {
            let preview: String = payload.chars().take(MAX_IMAGE_ERROR_CHARS).collect();
            return Err(format!("Image API HTTP {status}: {preview}"));
        }
        let value: Value = serde_json::from_str(&payload)
            .map_err(|error| format!("Image response is not JSON: {error}"))?;
        if let Some(encoded) = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("b64_json"))
            .and_then(Value::as_str)
            .filter(|encoded| !encoded.is_empty())
        {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| format!("Image base64 was invalid: {error}"))?;
            if bytes.len() > MAX_IMAGE_BYTES {
                return Err("Decoded image exceeded 64 MiB limit".into());
            }
            let media_type = detect_image_media_type(&bytes)
                .unwrap_or("image/png")
                .to_string();
            return Ok(ImageResult { bytes, media_type });
        }
        if let Some(url) = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("url"))
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
        {
            let response = self
                .client
                .get(url)
                .timeout(Duration::from_secs(120))
                .send()
                .map_err(|error| format!("Image download failed: {error}"))?;
            let status = response.status();
            if !status.is_success() {
                return Err(format!("Image download HTTP {status}"));
            }
            let header_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let bytes = read_bounded_bytes(response, MAX_IMAGE_BYTES)?;
            let media_type = detect_image_media_type(&bytes)
                .map(str::to_string)
                .or(header_type)
                .unwrap_or_else(|| "image/png".into());
            return Ok(ImageResult { bytes, media_type });
        }
        Err("Image response contained neither b64_json nor url".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::test_support;

    #[test]
    fn decodes_base64_image_response() {
        let png = [0x89, b'P', b'N', b'G'];
        let encoded = base64::engine::general_purpose::STANDARD.encode(png);
        let body = Box::leak(format!("{{\"data\":[{{\"b64_json\":\"{encoded}\"}}]}}").into_boxed_str());
        let server = test_support::serve_once(body, "application/json");
        let backend = OpenAiImageBackend::new(server.base_url, None);
        let result = backend
            .generate(&ImageRequest {
                model: "image-model".into(),
                prompt: "a tree".into(),
                size: "512x512".into(),
                output_format: "png".into(),
            })
            .expect("image response decodes");
        assert_eq!(result.bytes, png);
        assert_eq!(result.media_type, "image/png");
    }

    #[test]
    fn detects_jpeg_webp_gif_and_png_magic_bytes() {
        assert_eq!(detect_image_media_type(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
        assert_eq!(detect_image_media_type(b"GIF89a..."), Some("image/gif"));
        assert_eq!(detect_image_media_type(b"RIFF\x00\x00\x00\x00WEBP"), Some("image/webp"));
        assert_eq!(detect_image_media_type(b"unknown"), None);
    }
}
