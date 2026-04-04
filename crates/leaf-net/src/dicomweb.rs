//! DICOMweb client for modern HTTP-based PACS communication.

use leaf_core::config::PacsNodeConfig;
use leaf_core::error::{LeafError, LeafResult};
use reqwest::Client;
use serde_json::Value;
use tracing::debug;

/// DICOMweb client (QIDO-RS, WADO-RS, STOW-RS).
#[derive(Clone)]
pub struct DicomWebClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
}

impl DicomWebClient {
    pub fn new(config: &PacsNodeConfig) -> LeafResult<Self> {
        let base_url = config
            .dicomweb_url
            .clone()
            .ok_or_else(|| LeafError::Config("DICOMweb URL not configured".into()))?;

        let client = Client::builder().build().map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        Ok(Self {
            client,
            base_url,
            auth_token: config.auth_token.clone(),
        })
    }

    /// QIDO-RS: Search for studies.
    pub async fn search_studies(&self, params: &[(&str, &str)]) -> LeafResult<Vec<Value>> {
        let studies = self
            .get_json(&format!("{}/studies", self.base_url), params)
            .await?;

        debug!("QIDO-RS returned {} studies", studies.len());
        Ok(studies)
    }

    /// QIDO-RS: Search for series within a study.
    pub async fn search_series(
        &self,
        study_uid: &str,
        params: &[(&str, &str)],
    ) -> LeafResult<Vec<Value>> {
        self.get_json(
            &format!("{}/studies/{study_uid}/series", self.base_url),
            params,
        )
        .await
    }

    /// QIDO-RS: Search for instances within a series.
    pub async fn search_instances(
        &self,
        study_uid: &str,
        series_uid: &str,
        params: &[(&str, &str)],
    ) -> LeafResult<Vec<Value>> {
        self.get_json(
            &format!(
                "{}/studies/{study_uid}/series/{series_uid}/instances",
                self.base_url
            ),
            params,
        )
        .await
    }

    /// WADO-RS: Retrieve instance metadata.
    pub async fn get_metadata(
        &self,
        study_uid: &str,
        series_uid: &str,
        instance_uid: &str,
    ) -> LeafResult<Vec<Value>> {
        let url = format!(
            "{}/studies/{}/series/{}/instances/{}/metadata",
            self.base_url, study_uid, series_uid, instance_uid
        );
        let mut request = self.client.get(&url);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LeafError::DicomWeb {
                status,
                message: body,
            });
        }

        let metadata: Vec<Value> = response.json().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        Ok(metadata)
    }

    /// WADO-RS: Retrieve a rendered frame as PNG/JPEG bytes.
    pub async fn get_rendered_frame(
        &self,
        study_uid: &str,
        series_uid: &str,
        instance_uid: &str,
        frame: usize,
    ) -> LeafResult<Vec<u8>> {
        let url = format!(
            "{}/studies/{}/series/{}/instances/{}/rendered?frameNumber={}",
            self.base_url, study_uid, series_uid, instance_uid, frame
        );
        let mut request = self.client.get(&url).header("Accept", "image/png");

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LeafError::DicomWeb {
                status,
                message: body,
            });
        }

        let bytes = response.bytes().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        Ok(bytes.to_vec())
    }

    /// WADO-RS: Retrieve a raw DICOM instance as Part 10 bytes.
    ///
    /// Handles the standard `multipart/related` response envelope that DICOMweb
    /// servers return, extracting the first DICOM Part 10 payload automatically.
    pub async fn get_instance(
        &self,
        study_uid: &str,
        series_uid: &str,
        instance_uid: &str,
    ) -> LeafResult<Vec<u8>> {
        let url = format!(
            "{}/studies/{}/series/{}/instances/{}",
            self.base_url, study_uid, series_uid, instance_uid
        );
        let mut request = self.client.get(&url).header("Accept", "application/dicom");

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LeafError::DicomWeb {
                status,
                message: body,
            });
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();

        let bytes = response.bytes().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        if content_type
            .to_ascii_lowercase()
            .starts_with("multipart/related")
        {
            return extract_first_multipart_part(&bytes, &content_type);
        }

        Ok(bytes.to_vec())
    }

    /// WADO-RS: Retrieve all instances in a series as raw DICOM Part 10 bytes.
    ///
    /// Returns one `Vec<u8>` per instance extracted from the `multipart/related`
    /// response.  Falls back to treating the entire body as a single DICOM file
    /// when the Content-Type is not multipart.
    pub async fn get_series_instances(
        &self,
        study_uid: &str,
        series_uid: &str,
    ) -> LeafResult<Vec<Vec<u8>>> {
        let url = format!(
            "{}/studies/{}/series/{}",
            self.base_url, study_uid, series_uid
        );
        let mut request = self.client.get(&url).header("Accept", "application/dicom");

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LeafError::DicomWeb {
                status,
                message: body,
            });
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();

        let bytes = response.bytes().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        if content_type
            .to_ascii_lowercase()
            .starts_with("multipart/related")
        {
            return extract_all_multipart_parts(&bytes, &content_type);
        }

        Ok(vec![bytes.to_vec()])
    }

    async fn get_json(&self, url: &str, params: &[(&str, &str)]) -> LeafResult<Vec<Value>> {
        let mut request = self.client.get(url);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }

        let response = request.send().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LeafError::DicomWeb {
                status,
                message: body,
            });
        }

        response.json().await.map_err(|e| LeafError::DicomWeb {
            status: 0,
            message: e.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Multipart/related helpers
// ---------------------------------------------------------------------------

/// Extract the first DICOM Part 10 payload from a `multipart/related` body.
fn extract_first_multipart_part(body: &[u8], content_type: &str) -> LeafResult<Vec<u8>> {
    let boundary = extract_multipart_boundary(content_type)?;
    let parts = split_multipart_parts(body, &boundary);
    parts.into_iter().next().ok_or_else(|| LeafError::DicomWeb {
        status: 0,
        message: "multipart/related response contained no parts".into(),
    })
}

/// Extract all DICOM Part 10 payloads from a `multipart/related` body.
fn extract_all_multipart_parts(body: &[u8], content_type: &str) -> LeafResult<Vec<Vec<u8>>> {
    let boundary = extract_multipart_boundary(content_type)?;
    let parts = split_multipart_parts(body, &boundary);
    if parts.is_empty() {
        return Err(LeafError::DicomWeb {
            status: 0,
            message: "multipart/related response contained no parts".into(),
        });
    }
    Ok(parts)
}

fn extract_multipart_boundary(content_type: &str) -> LeafResult<String> {
    for segment in content_type.split(';') {
        let segment = segment.trim();
        if segment.to_ascii_lowercase().starts_with("boundary=") {
            let value = &segment["boundary=".len()..];
            return Ok(value.trim_matches('"').to_owned());
        }
    }
    Err(LeafError::DicomWeb {
        status: 0,
        message: "multipart Content-Type is missing the boundary parameter".into(),
    })
}

/// Split a multipart body on `boundary`, stripping per-part MIME headers.
fn split_multipart_parts(body: &[u8], boundary: &str) -> Vec<Vec<u8>> {
    let delimiter = format!("--{boundary}");
    let delimiter_bytes = delimiter.as_bytes();
    let header_sep = b"\r\n\r\n";

    let mut parts = Vec::new();
    let mut cursor = 0;

    while cursor < body.len() {
        // Locate the next boundary.
        let Some(boundary_pos) = find_subsequence(&body[cursor..], delimiter_bytes) else {
            break;
        };
        let after_boundary = cursor + boundary_pos + delimiter_bytes.len();

        // Check for closing delimiter (`--<boundary>--`).
        if body.get(after_boundary..after_boundary + 2) == Some(b"--") {
            break;
        }

        // Skip the CRLF after the boundary line.
        let part_start = if body.get(after_boundary..after_boundary + 2) == Some(b"\r\n") {
            after_boundary + 2
        } else {
            after_boundary
        };

        // Find the end-of-headers blank line inside this part.
        let data_start = if let Some(hdr_end) = find_subsequence(&body[part_start..], header_sep) {
            part_start + hdr_end + header_sep.len()
        } else {
            part_start
        };

        // The part data runs until the next boundary (preceded by `\r\n`).
        let next_boundary = format!("\r\n--{boundary}");
        let next_boundary_bytes = next_boundary.as_bytes();
        let data_end = if let Some(end) = find_subsequence(&body[data_start..], next_boundary_bytes)
        {
            cursor = data_start + end + next_boundary_bytes.len() - delimiter_bytes.len();
            data_start + end
        } else {
            cursor = body.len();
            body.len()
        };

        parts.push(body[data_start..data_end].to_vec());
    }

    parts
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
