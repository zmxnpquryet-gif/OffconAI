//! OpenAI-compatible image generation adapter used by the All-in-One edition.

use crate::contracts::{ImageGenerationBackend, ImageRequest, ImageResult};
use base64::Engine;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::io::Read;
use std::time::Duration;

const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_IMAGE_RESPONSE_BYTES: usize = 96 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
const MAX_BASE64_IMAGE_BYTES: usize = ((MAX_IMAGE_BYTES + 2) / 3) * 4;

pub struct OpenAiImageBackend {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

impl OpenAiImageBackend {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into().trim().trim_end_matches('/').to_string(),
            api_key: api_key
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty()),
            client: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("the HTTP client configuration is valid"),
        }
    }

    fn endpoint(&self) -> String {
        if self.base_url.ends_with("/images/generations") {
            return self.base_url.clone();
        }
        if self.base_url.ends_with("/v1") {
            format!("{}/images/generations", self.base_url)
        } else {
            format!("{}/v1/images/generations", self.base_url)
        }
    }
}

impl ImageGenerationBackend for OpenAiImageBackend {
    fn available(&self) -> bool {
        reqwest::Url::parse(&self.endpoint())
            .map(|url| matches!(url.scheme(), "http" | "https"))
            .unwrap_or(false)
    }

    fn generate(&self, request: &ImageRequest) -> Result<ImageResult, String> {
        if !self.available() {
            return Err("Image endpoint is not configured".into());
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
        if !status.is_success() {
            let (payload, truncated) = read_limited_body(response, MAX_ERROR_BODY_BYTES)?;
            let suffix = if truncated { "... (truncated)" } else { "" };
            return Err(format!(
                "Image API HTTP {status}: {}{suffix}",
                truncate_payload(&String::from_utf8_lossy(&payload))
            ));
        }
        let (payload, truncated) = read_limited_body(response, MAX_IMAGE_RESPONSE_BYTES)?;
        if truncated {
            return Err(format!(
                "Image response exceeded the {} MiB size limit",
                MAX_IMAGE_RESPONSE_BYTES / (1024 * 1024)
            ));
        }
        let value: Value = serde_json::from_slice(&payload)
            .map_err(|error| format!("Image response is not JSON: {error}"))?;
        if let Some(encoded) = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("b64_json"))
            .and_then(Value::as_str)
            .filter(|encoded| !encoded.is_empty())
        {
            if encoded.len() > MAX_BASE64_IMAGE_BYTES {
                return Err(format!(
                    "Generated image exceeded the {} MiB size limit",
                    MAX_IMAGE_BYTES / (1024 * 1024)
                ));
            }
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| format!("Image base64 was invalid: {error}"))?;
            if bytes.len() > MAX_IMAGE_BYTES {
                return Err(format!(
                    "Generated image exceeded the {} MiB size limit",
                    MAX_IMAGE_BYTES / (1024 * 1024)
                ));
            }
            return Ok(ImageResult {
                media_type: detect_image_media_type(&bytes).unwrap_or("image/png").into(),
                bytes,
            });
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
                let (payload, truncated) = read_limited_body(response, MAX_ERROR_BODY_BYTES)?;
                let suffix = if truncated { "... (truncated)" } else { "" };
                return Err(format!(
                    "Image download HTTP {status}: {}{suffix}",
                    truncate_payload(&String::from_utf8_lossy(&payload))
                ));
            }
            let media_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("image/png")
                .split(';')
                .next()
                .unwrap_or("image/png")
                .trim()
                .to_string();
            let (bytes, truncated) = read_limited_body(response, MAX_IMAGE_BYTES)?;
            if truncated {
                return Err(format!(
                    "Downloaded image exceeded the {} MiB size limit",
                    MAX_IMAGE_BYTES / (1024 * 1024)
                ));
            }
            let media_type = detect_image_media_type(&bytes)
                .unwrap_or(&media_type)
                .to_string();
            return Ok(ImageResult { bytes, media_type });
        }
        Err("Image response contained neither b64_json nor url".into())
    }
}

/// Read only a fixed prefix to keep provider responses and error pages bounded.
fn read_limited_body(
    response: reqwest::blocking::Response,
    limit: usize,
) -> Result<(Vec<u8>, bool), String> {
    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    response
        .take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Image response could not be read: {error}"))?;
    let truncated = bytes.len() > limit;
    bytes.truncate(limit);
    Ok((bytes, truncated))
}

/// Error payloads can be large HTML pages; keep the message readable.
fn truncate_payload(payload: &str) -> String {
    const LIMIT: usize = 500;
    let trimmed = payload.trim();
    if trimmed.len() <= LIMIT {
        return trimmed.to_string();
    }
    let mut end = LIMIT;
    while !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... (truncated)", trimmed[..end].trim_end())
}

fn detect_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        Some("image/jpeg")
    } else if bytes.len() >= 8
        && bytes[0] == 0x89
        && bytes[1] == b'P'
        && bytes[2] == b'N'
        && bytes[3] == b'G'
    {
        Some("image/png")
    } else if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        Some("image/gif")
    } else if bytes.len() >= 12
        && bytes.starts_with(b"RIFF")
        && bytes[8..].starts_with(b"WEBP")
    {
        Some("image/webp")
    } else {
        None
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
    }
}
