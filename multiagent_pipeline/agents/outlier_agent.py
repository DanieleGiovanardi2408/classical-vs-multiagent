"""OutlierAgent — quarto nodo del grafo multi-agent.

Responsabilità (dalla slide Reply):
    "Applies IsolationForest, LOF, or Z-score on the engineered features"

Implementa i tre modelli reali su sklearn, identici al notebook classico 04:
    - IsolationForest  (contamination=0.03, random_state=42)
    - LocalOutlierFactor (n_neighbors=20, contamination=0.03)
    - Z-score          (sulle BASELINE_FEATURES, già calcolato da BaselineAgent)

Ensemble pesata con gli stessi ENSEMBLE_WEIGHTS del classico.
Soglie data-driven (p97/p90) per coerenza con notebook 05.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from multiagent_pipeline.state import (
    AgentState,
    BASELINE_FEATURES,
    ENSEMBLE_WEIGHTS,
)

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stessi iperparametri del notebook classico 04
_CONTAMINATION  = 0.03
_N_NEIGHBORS    = 20
_RANDOM_STATE   = 42


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Seleziona le feature numeriche disponibili per i modelli ML.

    Usa BASELINE_FEATURES se presenti, altrimenti tutte le colonne numeriche
    escluse quelle di metadato/score già calcolati.
    """
    exclude = {"ZONA", "n_osservazioni_allarmi", "n_osservazioni_viag",
               "score_composito", "baseline_score", "baseline_flag"}
    exclude |= {c for c in df.columns if c.startswith("z_")}

    # Priorità: BASELINE_FEATURES presenti nel df
    bl_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    if bl_cols:
        cols = bl_cols
    else:
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude]

    X = df[cols].fillna(0.0)
    return X, cols


def run_outlier_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Applica IsolationForest, LOF e Z-score su df_baseline → ensemble score."""
    logger.info("OutlierAgent -- Avvio")
    started_at = time.perf_counter()

    try:
        df = state.get("df_baseline")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_baseline mancante: esegui prima BaselineAgent.")
        if df.empty:
            raise ValueError("df_baseline vuoto: impossibile stimare outlier.")

        out = df.copy()
        X, feat_cols = _get_feature_matrix(out)
        logger.info("Feature usate per ML: %d colonne — %s", len(feat_cols), feat_cols)

        # ── Normalizzazione ───────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── 1. IsolationForest ────────────────────────────────────────────────
        if_model = IsolationForest(
            contamination=_CONTAMINATION,
            random_state=_RANDOM_STATE,
            n_estimators=100,
        )
        # decision_function: più basso = più anomalo → invertiamo e normalizziamo
        if_raw = if_model.fit(X_scaled).decision_function(X_scaled)
        out["score_if"] = _minmax(pd.Series(-if_raw, index=out.index))
        logger.info("IsolationForest: score_if range [%.4f, %.4f]",
                    out["score_if"].min(), out["score_if"].max())

        # ── 2. LocalOutlierFactor ─────────────────────────────────────────────
        n_neighbors = min(_N_NEIGHBORS, len(out) - 1)
        lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=_CONTAMINATION,
        )
        lof_raw = lof_model.fit_predict(X_scaled)          # -1 anomalo, 1 normale
        lof_scores = lof_model.negative_outlier_factor_    # più negativo = più anomalo
        out["score_lof"] = _minmax(pd.Series(-lof_scores, index=out.index))
        logger.info("LOF: score_lof range [%.4f, %.4f]",
                    out["score_lof"].min(), out["score_lof"].max())

        # ── 3. Z-score (dalle colonne z_ prodotte da BaselineAgent) ──────────
        z_cols = [c for c in out.columns if c.startswith("z_")]
        if z_cols:
            z_proxy = out[z_cols].abs().mean(axis=1)
        else:
            # Fallback: z-score calcolato sul posto sulle feature
            z_proxy = pd.Series(
                np.abs(X_scaled).mean(axis=1), index=out.index
            )
        out["score_z"] = _minmax(z_proxy)
        logger.info("Z-score: score_z range [%.4f, %.4f]",
                    out["score_z"].min(), out["score_z"].max())

        # ── Ensemble pesata ───────────────────────────────────────────────────
        # AE non implementato: ripartiamo il suo peso su IF e LOF
        w_if  = ENSEMBLE_WEIGHTS["IF"]  + ENSEMBLE_WEIGHTS["AE"] * 0.5
        w_lof = ENSEMBLE_WEIGHTS["LOF"] + ENSEMBLE_WEIGHTS["AE"] * 0.5
        w_z   = ENSEMBLE_WEIGHTS["Z"]

        out["ensemble_score"] = (
            out["score_if"]  * w_if  +
            out["score_lof"] * w_lof +
            out["score_z"]   * w_z
        ).clip(0, 1)
        logger.info("Ensemble: range [%.4f, %.4f]",
                    out["ensemble_score"].min(), out["ensemble_score"].max())

        # ── Soglie data-driven (p97/p90) — identico al classico notebook 05 ──
        threshold_alta  = float(out["ensemble_score"].quantile(0.97))
        threshold_media = float(out["ensemble_score"].quantile(0.90))
        logger.info("Soglie data-driven: ALTA=%.4f (p97) | MEDIA=%.4f (p90)",
                    threshold_alta, threshold_media)

        out["risk_label"] = np.where(
            out["ensemble_score"] >= threshold_alta, "ALTA",
            np.where(out["ensemble_score"] >= threshold_media, "MEDIA", "NORMALE"),
        )

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "anomaly_results_live.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("OutlierAgent output salvato in: %s", saved_to)

        meta = {
            "n_alta"          : int((out["risk_label"] == "ALTA").sum()),
            "n_media"         : int((out["risk_label"] == "MEDIA").sum()),
            "n_normale"       : int((out["risk_label"] == "NORMALE").sum()),
            "soglia_alta"     : threshold_alta,
            "soglia_media"    : threshold_media,
            "threshold_method": "data-driven (p97/p90)",
            "metodo_ensemble" : "IF + LOF + Z-score (sklearn reali)",
            "feature_cols"    : feat_cols,
            "n_features"      : len(feat_cols),
            "saved_to"        : saved_to,
            "top_rotte"       : (
                out.sort_values("ensemble_score", ascending=False)
                .head(10)[["ROTTA", "ensemble_score", "risk_label"]]
                .to_dict(orient="records")
            ),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "OutlierAgent ✓ Completato — ALTA=%d MEDIA=%d NORMALE=%d (%.2fs)",
            meta["n_alta"], meta["n_media"], meta["n_normale"], meta["elapsed_s"],
        )
        return {**state, "df_anomalies": out, "anomaly_meta": meta}

    except Exception as e:
        logger.error("OutlierAgent ✗ Errore: %s", e)
        return {
            **state,
            "df_anomalies": None,
            "anomaly_meta": {
                "error"       : str(e),
                "user_message": "Outlier detection fallita: verifica output baseline e filtri.",
                "elapsed_s"   : round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    from multiagent_pipeline.agents.feature_agent import run_feature_agent
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.tools.data_tools import load_last_perimeter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _perimeter = load_last_perimeter() or {"anno": 2024}
    print(f"  Perimetro: {_perimeter}")
    s: AgentState = {"perimeter": _perimeter}
    s = data_agent_node(s)
    s = run_feature_agent(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    print("\n=== RISULTATO OutlierAgent ===")
    am = s["anomaly_meta"]
    print(f"  ALTA={am['n_alta']} | MEDIA={am['n_media']} | NORMALE={am['n_normale']}")
    print(f"  soglia_alta={am['soglia_alta']:.4f} | soglia_media={am['soglia_media']:.4f}")
    print(f"  metodo: {am['metodo_ensemble']}")
    print(f"  features: {am['n_features']} colonne")
    print(f"  elapsed: {am['elapsed_s']}s")
    print(f"  top rotte: {am['top_rotte'][:3]}")
