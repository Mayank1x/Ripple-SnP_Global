"""Isotonic Regression Calibration for Risk Impact Scores.

Maps raw_impact (0.0 to 1.0) to realized market abnormal moves (|CAR| + volatility shift)
using a strict time-based split (train on earlier events, test on later events).
Produces calibrated impact scores in 1.0 to 10.0.
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from src.ingest.helpers import load_config
from src.scoring.event_study import compute_event_study
from src.scoring.schema import Event, Signal

CALIBRATION_DIR = Path("data/calibration")


@dataclass
class CalibrationMetadata:
    """Audit metadata attached to calibrated signals."""

    model_name: str
    version: str
    calibration_date: str
    train_samples: int
    test_samples: int
    test_spearman_rho: float
    test_p_value: float
    min_predicted_move: float
    max_predicted_move: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationMetadata:
        return cls(**data)


class ImpactCalibrator:
    """Calibrates raw NLP impact scores to empirical market moves via Isotonic Regression."""

    def __init__(self) -> None:
        self.model: IsotonicRegression | None = None
        self.metadata: CalibrationMetadata | None = None
        self.y_min: float = 0.005
        self.y_max: float = 0.080
        self.bin_thresholds: list[float] = []

    def fit(
        self,
        event_pairs: list[dict[str, Any]],
        time_split_ratio: float = 0.70,
    ) -> dict[str, Any]:
        """Fit isotonic regression on time-split event study data.

        Args:
            event_pairs: List of dicts with 'raw_impact', 'realized_move', and 'date'.
            time_split_ratio: Fraction of earliest events reserved for training.

        Returns:
            Dictionary containing backtest metrics on the out-of-sample test split.
        """
        if len(event_pairs) < 10:
            raise ValueError(f"Need at least 10 event pairs to fit calibration, got {len(event_pairs)}")

        # 1. Sort strictly by date (prevent look-ahead bias)
        sorted_pairs = sorted(event_pairs, key=lambda x: str(x.get("date", "")))

        X_all = np.array([float(p["raw_impact"]) for p in sorted_pairs], dtype=np.float64)
        Y_all = np.array([float(p["realized_move"]) for p in sorted_pairs], dtype=np.float64)

        n_total = len(sorted_pairs)
        n_train = max(7, int(n_total * time_split_ratio))
        n_test = n_total - n_train

        X_train, Y_train = X_all[:n_train], Y_all[:n_train]
        X_test, Y_test = X_all[n_train:], Y_all[n_train:]

        # 2. Fit Isotonic Regression (monotonically non-decreasing)
        self.model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.model.fit(X_train, Y_train)

        # 3. Predict on Out-of-Sample Test Split
        pred_test = self.model.predict(X_test)
        pred_train = self.model.predict(X_train)

        # Min and max fitted bounds for linear scaling to 1.0 - 10.0
        self.y_min = float(np.min(pred_train))
        self.y_max = float(np.max(pred_train))
        if abs(self.y_max - self.y_min) < 1e-6:
            self.y_max = self.y_min + 0.05

        # 4. Out-of-Sample Statistical Evaluation
        # Spearman Rank Correlation
        if len(pred_test) >= 3 and np.std(pred_test) > 1e-8:
            rho, p_val = stats.spearmanr(X_test, Y_test)
            rho = float(rho) if not np.isnan(rho) else 0.0
            p_val = float(p_val) if not np.isnan(p_val) else 1.0
        else:
            rho, p_val = 0.0, 1.0

        mae_test = float(np.mean(np.abs(Y_test - pred_test))) if n_test > 0 else 0.0
        rmse_test = float(np.sqrt(np.mean((Y_test - pred_test) ** 2))) if n_test > 0 else 0.0

        # Bin thresholds across 10 deciles
        self.bin_thresholds = [
            float(self.y_min + (self.y_max - self.y_min) * (i / 10.0)) for i in range(1, 11)
        ]

        self.metadata = CalibrationMetadata(
            model_name="IsotonicRegression",
            version="1.0.0",
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
            train_samples=n_train,
            test_samples=n_test,
            test_spearman_rho=round(rho, 4),
            test_p_value=round(p_val, 4),
            min_predicted_move=round(self.y_min, 6),
            max_predicted_move=round(self.y_max, 6),
        )

        # 5. Persist calibration model
        self.save()

        return {
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "spearman_rho": round(rho, 4),
            "p_value": round(p_val, 4),
            "mae_test": round(mae_test, 6),
            "rmse_test": round(rmse_test, 6),
            "y_min": round(self.y_min, 6),
            "y_max": round(self.y_max, 6),
            "test_actuals": Y_test.tolist(),
            "test_predictions": pred_test.tolist(),
            "test_raw_impacts": X_test.tolist(),
        }

    def predict_score(self, raw_impact: float) -> tuple[float, dict[str, Any]]:
        """Map raw_impact (0.0 to 1.0) to calibrated impact score (1.0 to 10.0).

        Returns:
            (calibrated_score, metadata_dict)
        """
        raw_clamped = max(0.0, min(1.0, float(raw_impact)))

        if self.model is None:
            self.load()

        if self.model is not None:
            try:
                pred_move = float(self.model.predict([raw_clamped])[0])
                # Linear binning between y_min and y_max to 1.0 - 10.0
                norm_pos = (pred_move - self.y_min) / (self.y_max - self.y_min)
                norm_pos = max(0.0, min(1.0, norm_pos))
                score = round(1.0 + 9.0 * norm_pos, 1)
                meta = self.metadata.to_dict() if self.metadata else {}
                meta["predicted_abnormal_move"] = round(pred_move, 5)
                return score, meta
            except Exception:
                pass

        # Fallback linear mapping if no calibration loaded
        fallback_score = round(1.0 + 9.0 * raw_clamped, 1)
        return fallback_score, {"status": "uncalibrated_linear_fallback"}

    def save(self, filepath: Path | None = None) -> None:
        """Save fitted calibration model parameters and metadata to JSON."""
        target_dir = filepath.parent if filepath else CALIBRATION_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = filepath or (target_dir / "impact_calibration.json")

        if self.model is None or self.metadata is None:
            return

        payload = {
            "metadata": self.metadata.to_dict(),
            "y_min": self.y_min,
            "y_max": self.y_max,
            "bin_thresholds": self.bin_thresholds,
            "X_thresholds": self.model.X_thresholds_.tolist() if hasattr(self.model, "X_thresholds_") else [],
            "y_thresholds": self.model.y_thresholds_.tolist() if hasattr(self.model, "y_thresholds_") else [],
        }

        with open(target_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load(self, filepath: Path | None = None) -> bool:
        """Load calibration model from JSON."""
        target_file = filepath or (CALIBRATION_DIR / "impact_calibration.json")
        if not target_file.exists():
            return False

        try:
            with open(target_file, encoding="utf-8") as fh:
                data = json.load(fh)

            self.metadata = CalibrationMetadata.from_dict(data["metadata"])
            self.y_min = float(data.get("y_min", 0.005))
            self.y_max = float(data.get("y_max", 0.080))
            self.bin_thresholds = data.get("bin_thresholds", [])

            x_thresh = np.array(data["X_thresholds"], dtype=np.float64)
            y_thresh = np.array(data["y_thresholds"], dtype=np.float64)

            self.model = IsotonicRegression(out_of_bounds="clip", increasing=True)
            self.model.X_thresholds_ = x_thresh
            self.model.y_thresholds_ = y_thresh
            return True
        except Exception:
            return False


# Global singleton instance
_GLOBAL_CALIBRATOR = ImpactCalibrator()


def get_calibrator() -> ImpactCalibrator:
    """Return initialized global ImpactCalibrator."""
    global _GLOBAL_CALIBRATOR
    if _GLOBAL_CALIBRATOR.model is None:
        _GLOBAL_CALIBRATOR.load()
    return _GLOBAL_CALIBRATOR
