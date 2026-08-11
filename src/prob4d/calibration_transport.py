"""Source-only calibration-transport support certificates.

The module summarizes complete source objects or acquisition sessions with robust
feature quantiles, calibrates a nearest-source nonconformity threshold from those
independent units only, and evaluates target-prefix feature support without
opening target residuals or downstream physical outcomes.
"""

from ._calibration_transport_common import (
    CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
    CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA,
    CALIBRATION_TRANSPORT_FEATURE_CONTRACT_SCHEMA,
    CALIBRATION_TRANSPORT_MODEL_SCHEMA,
    CALIBRATION_TRANSPORT_VERSION,
    CalibrationTransportPolicyV1,
    CalibrationTransportUnitV1,
    calibration_transport_feature_contract_id,
)
from ._calibration_transport_evidence import (
    CalibrationTransportEvidenceV1,
    evaluate_calibration_transport,
)
from ._calibration_transport_group import CalibrationTransportGroupResultV1
from ._calibration_transport_io import (
    load_calibration_transport_evidence,
    load_calibration_transport_model,
    save_calibration_transport_evidence,
    save_calibration_transport_model,
)
from ._calibration_transport_model import (
    CalibrationTransportModelV1,
    fit_calibration_transport_model,
)

__all__ = [
    "CALIBRATION_TRANSPORT_CLAIM_BOUNDARY",
    "CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA",
    "CALIBRATION_TRANSPORT_FEATURE_CONTRACT_SCHEMA",
    "CALIBRATION_TRANSPORT_MODEL_SCHEMA",
    "CALIBRATION_TRANSPORT_VERSION",
    "CalibrationTransportEvidenceV1",
    "CalibrationTransportGroupResultV1",
    "CalibrationTransportModelV1",
    "CalibrationTransportPolicyV1",
    "CalibrationTransportUnitV1",
    "calibration_transport_feature_contract_id",
    "evaluate_calibration_transport",
    "fit_calibration_transport_model",
    "load_calibration_transport_evidence",
    "load_calibration_transport_model",
    "save_calibration_transport_evidence",
    "save_calibration_transport_model",
]
