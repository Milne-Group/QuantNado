"""Pydantic models for peak calling method options."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QuantileOptions(BaseModel):
    """Options for quantile-threshold peak calling."""

    tilesize: int = Field(128, gt=0, description="Size of genomic tiles in bp")
    window_overlap: int = Field(8, ge=0, description="Overlap between adjacent windows in bp")
    quantile: float = Field(0.98, gt=0, lt=1, description="Quantile threshold for peak calling")
    merge: bool = Field(True, description="Merge overlapping and adjacent peaks after calling")
    merge_distance: int = Field(150, ge=0, description="Maximum distance (bp) between peaks to merge (when merge=True)")
    n_workers: int = Field(1, gt=0, description="Number of parallel workers for chromosome processing")

    @model_validator(mode="after")
    def _check_overlap(self):
        if self.window_overlap >= self.tilesize:
            raise ValueError("window_overlap must be < tilesize")
        return self


class SeacrOptions(BaseModel):
    """Options for SEACR-style AUC island calling."""

    control_sample_name: str | None = Field(None, description="Name of control sample for normalization")
    fdr_threshold: float = Field(0.01, gt=0, lt=1, description="Numeric FDR threshold (0–1)")
    norm: Literal["norm", "non"] = Field("non", description='Normalize control to experimental signal')
    stringency: Literal["stringent", "relaxed"] = Field("stringent", description="Peak of AUC curve or knee")


class LanceotronOptions(BaseModel):
    """Options for LanceOtron ML-based peak calling."""

    score_threshold: float = Field(0.5, ge=0, le=1, description="Minimum overall_classification score")
    smooth_window: int = Field(400, gt=0, description="Rolling mean window for candidate detection (bp)")
    batch_size: int = Field(512, gt=0, description="Inference batch size")


PeakOptions = QuantileOptions | SeacrOptions | LanceotronOptions
