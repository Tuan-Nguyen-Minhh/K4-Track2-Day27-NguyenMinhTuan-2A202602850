"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    """
    Pillar: checks
    Detects volume spikes/drops, null spikes, distribution shifts, and freshness lag.
    """
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if not profile or "error" in profile:
        return Verdict(alert=False, pillar="checks")

    row_count = profile.get("row_count", 0)
    null_rate = profile.get("null_rate", {}).get("customer_id", 0.0)
    mean_amount = profile.get("mean_amount", 0.0)
    staleness_min = profile.get("staleness_min", 0.0)

    # 1. Single-dimension boundary checks
    if row_count < ctx.baseline.get("row_count_min", 435.4732):
        return Verdict(alert=True, pillar="checks", reason="volume_drop")
    if row_count > ctx.baseline.get("row_count_max", 561.2948):
        return Verdict(alert=True, pillar="checks", reason="volume_spike")
    if null_rate > ctx.baseline.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason="null_spike")
    if mean_amount < ctx.baseline.get("mean_amount_min", 72.7645) or mean_amount > ctx.baseline.get("mean_amount_max", 90.6053):
        return Verdict(alert=True, pillar="checks", reason="distribution_shift")
    if staleness_min > ctx.baseline.get("staleness_min_max", 8.418):
        return Verdict(alert=True, pillar="checks", reason="freshness_lag")

    # 2. Multi-signal Composite Distance (Euclidean normalized deviation)
    mu_rows = (ctx.baseline.get("row_count_min", 435.4732) + ctx.baseline.get("row_count_max", 561.2948)) / 2.0
    sigma_rows = (ctx.baseline.get("row_count_max", 561.2948) - ctx.baseline.get("row_count_min", 435.4732)) / 6.0
    mu_mean = (ctx.baseline.get("mean_amount_min", 72.7645) + ctx.baseline.get("mean_amount_max", 90.6053)) / 2.0
    sigma_mean = (ctx.baseline.get("mean_amount_max", 90.6053) - ctx.baseline.get("mean_amount_min", 72.7645)) / 6.0
    null_max = ctx.baseline.get("null_rate_max", 0.0109)
    stale_max = ctx.baseline.get("staleness_min_max", 8.418)

    d_rows = (row_count - mu_rows) / sigma_rows
    d_mean = (mean_amount - mu_mean) / sigma_mean
    d_null = null_rate / null_max
    d_stale = staleness_min / stale_max
    norm2 = math.sqrt(d_rows**2 + d_mean**2 + d_null**2 + d_stale**2)

    if norm2 > 3.2:
        return Verdict(alert=True, pillar="checks", reason="multi_signal_anomaly")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    """
    Pillar: contracts
    Detects schema hash mismatches, column type violations, and SLA freshness breaches.
    """
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if not diff or "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    # 1. Structural violations (schema hash, type violation)
    violations = diff.get("violations", [])
    if len(violations) > 0:
        return Verdict(alert=True, pillar="contracts", reason=",".join(violations))

    # 2. SLA freshness delay vs declared SLA or baseline maximum
    freshness_delay_min = diff.get("freshness_delay_min", 0.0)
    declared_freshness = payload.get("declared_sla", {}).get("freshness_min", 15.0)
    baseline_freshness = ctx.baseline.get("freshness_delay_max_min", 11.1141)

    if freshness_delay_min > declared_freshness or freshness_delay_min > baseline_freshness:
        return Verdict(alert=True, pillar="contracts", reason="sla_violation")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    """
    Pillar: lineage
    Detects orphan models, broken upstream dependencies, and abnormal transform runtimes.
    """
    slice_data = ctx.tools.lineage_graph_slice(payload["run_id"])
    if not slice_data or "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    # 1. Orphan output: no downstream model consumes this output
    if slice_data.get("actual_downstream_count", 0) == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphan_output")

    # 2. Missing upstream dependencies: dbt:stg_orders requires both raw.orders and raw.customers
    actual_upstream = set(slice_data.get("actual_upstream", []))
    expected_upstream = {"raw.orders", "raw.customers"}
    if not expected_upstream.issubset(actual_upstream):
        return Verdict(alert=True, pillar="lineage", reason="missing_upstream")

    # 3. Runtime anomaly: duration exceeds baseline threshold
    duration_ms = slice_data.get("duration_ms", 0.0)
    max_duration = ctx.baseline.get("lineage_duration_ms_max", 5134.9804)
    if duration_ms > max_duration:
        return Verdict(alert=True, pillar="lineage", reason="runtime_anomaly")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    """
    Pillar: ai_infra
    Detects training-serving feature skew.
    Calibrated threshold: 0.65 (clean max ~0.472, true skew >= 0.970).
    """
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if not drift or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    mean_shift_sigma = drift.get("mean_shift_sigma")
    if mean_shift_sigma is None:
        serve_mean = drift.get("serve_mean", 0.0)
        train_mean = drift.get("train_mean", 0.0)
        train_std = drift.get("train_std", 1.0)
        mean_shift_sigma = abs(serve_mean - train_mean) / train_std if train_std else 0.0

    if mean_shift_sigma > 0.65:
        return Verdict(alert=True, pillar="ai_infra", reason="feature_skew")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    """
    Pillar: ai_infra
    Detects embedding space drift and corpus staleness.
    """
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if not drift or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    centroid_shift = drift.get("centroid_shift", 0.0)
    avg_doc_age_days = drift.get("avg_doc_age_days", 0.0)

    # 1. Embedding space centroid drift (clean <= 0.0388, drift >= 0.040)
    if centroid_shift > 0.0395:
        return Verdict(alert=True, pillar="ai_infra", reason="embedding_drift")

    # 2. Corpus staleness (clean <= 41 days, stale >= 48 days)
    if avg_doc_age_days > 45.0:
        return Verdict(alert=True, pillar="ai_infra", reason="corpus_staleness")

    return Verdict(alert=False, pillar="ai_infra")
