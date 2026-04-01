//! DICOMweb client for modern HTTP-based PACS communication.

use leaf_core::config::PacsNodeConfig;
use leaf_core::error::{LeafError, LeafResult};
use reqwest::Client;
use serde_json::Value;
use tracing::debug;

/// DICOMweb client (QIDO-RS, WADO-RS, STOW-RS).
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
