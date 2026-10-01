from __future__ import annotations

from dataclasses import dataclass

from paint_inspection.core.types import Verdict


@dataclass(slots=True)
class FusionDecision:
    verdict: Verdict
    score: float
    reasons: list[str]


def fuse(anomaly_score: float, anomaly_area_percent: float, colour_score: float,
         texture_score: float, glare_percent: float, defect_count: int,
         config: dict, glare_strategy: str = "REDUCE_ANOMALY_WEIGHT") -> FusionDecision:
    weights = [float(config.get("anomaly_weight", .5)), float(config.get("colour_weight", .25)), float(config.get("texture_weight", .25))]
    agreement = sum(v >= .55 for v in (anomaly_score, colour_score, texture_score))
    anomaly_evidence = anomaly_score * min(1.0, anomaly_area_percent / .1) if defect_count else 0.0
    score = weights[0] * anomaly_evidence + weights[1] * colour_score + weights[2] * texture_score
    reasons = [f"{agreement} inspection branches agree"]
    if agreement >= 2:
        score = min(1.0, score + .18); reasons.append("multi-signal corroboration")
    if glare_strategy == "REDUCE_ANOMALY_WEIGHT" and glare_percent > 0:
        reduction = min(.25, glare_percent / 100 * .5); score *= 1 - reduction
        reasons.append(f"glare confidence reduction {reduction:.2f}")
    fail, passed = float(config.get("fail_threshold", .67)), float(config.get("pass_threshold", .30))
    if defect_count and score >= fail and agreement >= 2:
        verdict = Verdict.FAIL
    elif score <= passed and (defect_count == 0 or agreement == 0):
        verdict = Verdict.PASS
    else:
        verdict = Verdict.UNCERTAIN
    return FusionDecision(verdict, float(score), reasons)

