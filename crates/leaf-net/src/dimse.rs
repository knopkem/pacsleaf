//! DIMSE SCU client for traditional PACS communication.

use dicom_toolkit_net::{
    Association, AssociationConfig, FindRequest, PresentationContextRq,
    c_echo, c_find,
};
use leaf_core::config::PacsNodeConfig;
use leaf_core::error::{LeafError, LeafResult};
use tracing::{debug, info};

/// DIMSE client for communicating with a remote PACS node.
pub struct DimseClient {
    config: PacsNodeConfig,
    local_ae: String,
}

impl DimseClient {
    pub fn new(config: PacsNodeConfig, local_ae: String) -> Self {
        Self { config, local_ae }
    }

    fn assoc_config(&self) -> AssociationConfig {
        AssociationConfig {
            local_ae_title: self.local_ae.clone(),
            ..AssociationConfig::default()
        }
    }

    /// Test connectivity with C-ECHO.
    pub async fn echo(&self) -> LeafResult<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let verification_ctx = PresentationContextRq {
            id: 1,
            abstract_syntax: "1.2.840.10008.1.1".to_string(),
            transfer_syntaxes: vec!["1.2.840.10008.1.2".to_string()],
        };

        let mut assoc = Association::request(
            &addr,
            &self.config.ae_title,
            &self.local_ae,
            &[verification_ctx],
            &self.assoc_config(),
        )
        .await
        .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        c_echo(&mut assoc, 1)
            .await
            .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        assoc
            .release()
            .await
            .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        info!("C-ECHO succeeded to {}", self.config.name);
        Ok(())
    }

    /// Query studies via C-FIND at Study level.
    pub async fn find_studies(&self, query_bytes: Vec<u8>) -> LeafResult<Vec<Vec<u8>>> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let study_root_find = "1.2.840.10008.5.1.4.1.2.2.1";
        let find_ctx = PresentationContextRq {
            id: 1,
            abstract_syntax: study_root_find.to_string(),
            transfer_syntaxes: vec!["1.2.840.10008.1.2".to_string()],
        };

        let mut assoc = Association::request(
            &addr,
            &self.config.ae_title,
            &self.local_ae,
            &[find_ctx],
            &self.assoc_config(),
        )
        .await
        .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        let request = FindRequest {
            sop_class_uid: study_root_find.to_string(),
            query: query_bytes,
            context_id: 1,
            priority: 0,
        };

        let results = c_find(&mut assoc, request)
            .await
            .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        assoc
            .release()
            .await
            .map_err(|e| LeafError::DicomNetwork(e.to_string()))?;

        debug!(
            "C-FIND returned {} results from {}",
            results.len(),
            self.config.name
        );
        Ok(results)
    }
}
