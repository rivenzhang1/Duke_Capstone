from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import timedelta
import re

import numpy as np
import pandas as pd

START = "<!-- ARIMAX_ADDON:START -->"
END = "<!-- ARIMAX_ADDON:END -->"
H_START = "<!-- HORIZON_COMPARISON:START -->"
H_END = "<!-- HORIZON_COMPARISON:END -->"

DEFAULT_V3_SOURCE = Path(__file__).resolve().parents[1] / "_data" / "v3" / "stay_dates.parquet"
DEFAULT_ARIMAX_HORIZONS = [7, 10, 30, 90, 180]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def detect_horizon_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    for c in ("exact_horizon", "horizon", "fh", "lead", "days_ahead"):
        if c in df.columns:
            return c
    raise ValueError(f"No horizon column found. Available: {list(df.columns)}")


def compute_metrics(df: pd.DataFrame, case_name: str, horizon_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = {"y_true", "y_pred", horizon_col}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{case_name}: missing columns {sorted(missing)}")

    d = df.copy()
    d["err"] = d["y_pred"] - d["y_true"]
    d["abs_err"] = d["err"].abs()
    d["ape"] = np.where(d["y_true"] != 0, d["abs_err"] / d["y_true"].abs(), np.nan)

    def agg(x: pd.DataFrame) -> dict:
        denom = x["y_true"].abs().sum()
        wmape = (x["abs_err"].sum() / denom) if denom != 0 else np.nan
        return {
            "WMAPE": f"{wmape:.4f}" if pd.notna(wmape) else "nan",
            "MAE": f"{x['abs_err'].mean():.3f}",
            "Bias": f"{x['err'].mean():.3f}",
            "MAPE": f"{x['ape'].mean(skipna=True):.4f}",
            "N": int(len(x)),
        }

    overall_vals = agg(d)
    overall = pd.DataFrame([{"Case": case_name, **overall_vals}])

    rows = []
    for h, g in d.groupby(horizon_col):
        rows.append({"Horizon": h, "Case": case_name, **agg(g)})
    by_h = pd.DataFrame(rows).sort_values(["Horizon", "Case"])
    return overall, by_h


def upsert_section(text: str, block: str, start_tag: str, end_tag: str) -> str:
    if start_tag in text and end_tag in text:
        s = text.index(start_tag)
        e = text.index(end_tag) + len(end_tag)
        return text[:s] + block + text[e:]
    return text.rstrip() + "\n\n" + block + "\n"


def upsert(text: str, block: str) -> str:
    # backward-compatible wrapper for ARIMAX block
    return upsert_section(text, block, START, END)


def _is_placeholder_path(p: Path | None) -> bool:
    if p is None:
        return False
    s = str(p)
    return s.startswith("/path/to/") or "REPLACE_ME" in s


def _pick_col(cols: list[str], aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _pick_col_fuzzy(cols: list[str], keywords: list[str]) -> str | None:
    # pick first column containing any keyword
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keywords):
            return c
    return None


def _column_mapping(cols: list[str]) -> dict[str, str] | None:
    # exact/alias first
    prop = _pick_col(cols, ["property_id", "property", "hotel_id", "propertyid", "pid"])
    origin = _pick_col(cols, ["origin_date", "origin", "forecast_origin", "as_of_date", "origin_dt"])
    stay = _pick_col(cols, ["stay_date", "target_date", "date", "target_dt", "stay_dt", "arrival_date", "ds"])
    y = _pick_col(cols, ["y_true", "actual", "actuals", "y", "target", "rooms_sold", "final_rooms_sold", "observed"])

    # fuzzy fallback
    prop = prop or _pick_col_fuzzy(cols, ["property", "hotel"])
    origin = origin or _pick_col_fuzzy(cols, ["origin", "as_of", "asof", "snapshot"])
    # avoid picking origin again for stay
    stay = stay or next(
        (c for c in cols if c != origin and any(k in c.lower() for k in ["stay", "target", "arrival", "label", "date", "ds"])),
        None,
    )
    y = y or _pick_col_fuzzy(cols, ["y_true", "actual", "target", "rooms_sold", "observed", "label"])

    mapping: dict[str, str] = {}
    if prop:
        mapping[prop] = "property_id"
    if origin:
        mapping[origin] = "origin_date"
    if stay:
        mapping[stay] = "stay_date"
    if y:
        mapping[y] = "y_true"

    # minimum required now: property, origin, y
    # stay can be derived later from horizon
    if {"property_id", "origin_date", "y_true"}.issubset(set(mapping.values())):
        return mapping
    return None


def _read_preview(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, nrows=1)
    if suf == ".parquet":
        return pd.read_parquet(path).head(1)
    if suf == ".feather":
        return pd.read_feather(path).head(1)
    raise ValueError(f"Unsupported file type: {path}")


def _read_full(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")


def _has_required_cols(path: Path) -> bool:
    try:
        cols = list(_read_preview(path).columns)
    except Exception:
        return False
    return _column_mapping(cols) is not None


def _normalize_input_rows(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _column_mapping(list(df.columns))
    if mapping is None:
        raise ValueError(
            "Could not map required columns. Need aliases for property/origin/y_true. "
            f"Available: {list(df.columns)}"
        )

    out = df.rename(columns=mapping).copy()

    if "origin_date" not in out.columns or "property_id" not in out.columns or "y_true" not in out.columns:
        raise ValueError("Missing required mapped columns: property_id, origin_date, y_true")

    out["origin_date"] = pd.to_datetime(out["origin_date"])

    if "stay_date" not in out.columns:
        # derive from horizon if available
        hz_col = None
        for c in ("exact_horizon", "horizon", "fh", "lead", "days_ahead"):
            if c in out.columns:
                hz_col = c
                break
        if hz_col is None:
            raise ValueError(
                "Missing stay_date and cannot derive it (no horizon column found). "
                f"Available: {list(out.columns)}"
            )
        out["stay_date"] = out["origin_date"] + pd.to_timedelta(pd.to_numeric(out[hz_col], errors="coerce"), unit="D")
    else:
        out["stay_date"] = pd.to_datetime(out["stay_date"])

    out = out.dropna(subset=["property_id", "origin_date", "stay_date", "y_true"]).copy()
    return out


def resolve_input_rows(run_dir: Path, input_rows: Path | None) -> Path:
    if _is_placeholder_path(input_rows):
        raise FileNotFoundError(
            f"Placeholder path used: {input_rows}. Provide a real --input-rows path."
        )
    if input_rows is not None:
        if not input_rows.exists():
            raise FileNotFoundError(f"--input-rows not found: {input_rows}")
        if not _has_required_cols(input_rows):
            cols = list(_read_preview(input_rows).columns)
            raise FileNotFoundError(
                f"--input-rows exists but required columns not mappable. Columns: {cols}"
            )
        return input_rows

    # recursive search in common row-level formats
    candidates = []
    for ext in ("*.csv", "*.parquet", "*.feather"):
        candidates.extend(sorted(run_dir.rglob(ext)))

    good = [p for p in candidates if _has_required_cols(p)]
    if good:
        return good[0]

    sample = [str(c) for c in candidates[:30]]
    raise FileNotFoundError(
        "Could not auto-find row-level input in run-dir. "
        "Need mappable columns for property_id, origin_date, stay_date, y_true. "
        f"Scanned {len(candidates)} files. Sample: {sample}"
    )


def _debug_summary(df: pd.DataFrame, label: str) -> None:
    print(f"[DEBUG] {label}: rows={len(df):,}, cols={len(df.columns)}")
    print(f"[DEBUG] columns: {list(df.columns)}")

    for c in ("property_id", "origin_date", "stay_date", "y_true"):
        if c in df.columns:
            na = int(df[c].isna().sum())
            print(f"[DEBUG] {c}: nulls={na:,}")

    if "property_id" in df.columns:
        print(f"[DEBUG] unique properties: {df['property_id'].nunique():,}")
    if "origin_date" in df.columns:
        print(f"[DEBUG] unique origins: {df['origin_date'].nunique():,}")
        print(f"[DEBUG] origin_date range: {df['origin_date'].min()} -> {df['origin_date'].max()}")
    if "stay_date" in df.columns:
        print(f"[DEBUG] stay_date range: {df['stay_date'].min()} -> {df['stay_date'].max()}")

    key_cols = [c for c in ("property_id", "origin_date", "stay_date") if c in df.columns]
    if len(key_cols) == 3:
        dups = int(df.duplicated(key_cols).sum())
        print(f"[DEBUG] duplicate keys ({key_cols}): {dups:,}")


def _parse_horizons(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("No horizons provided.")
    vals = sorted(set(vals))
    if any(v <= 0 for v in vals):
        raise ValueError(f"Horizons must be positive integers. Got: {vals}")
    return vals


def _normalize_v3_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize v3 raw daily series to columns:
      property_id, date, y_true
    Requires a rooms-sold-like column.
    """
    cols = list(df.columns)
    prop = _pick_col(cols, ["property_id", "property", "hotel_id", "pid"]) or _pick_col_fuzzy(cols, ["property", "hotel"])
    date = _pick_col(cols, ["date", "stay_date", "ds", "day", "target_date"]) or _pick_col_fuzzy(cols, ["date", "stay", "ds"])

    # strict: require rooms_sold-style column
    rooms = _pick_col(cols, ["rooms_sold", "final_rooms_sold", "rooms", "sold_rooms"])
    if rooms is None:
        rooms = _pick_col_fuzzy(cols, ["rooms_sold", "final_rooms", "sold"])

    if not (prop and date and rooms):
        raise ValueError(
            "ARIMAX source-v3 must include property/date/rooms_sold-like columns. "
            f"Available: {cols}"
        )

    out = df.rename(columns={prop: "property_id", date: "date", rooms: "y_true"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out = out.dropna(subset=["property_id", "date", "y_true"]).copy()
    return out.sort_values(["property_id", "date"], kind="stable")


def _validate_arimax_base(base: pd.DataFrame) -> None:
    need = {"property_id", "origin_date", "stay_date", "y_true"}
    miss = need - set(base.columns)
    if miss:
        raise ValueError(f"ARIMAX base missing required columns: {sorted(miss)}")
    if base.empty:
        raise RuntimeError("ARIMAX base dataset is empty.")


def run_arimax_if_needed(
    run_dir: Path,
    input_rows: Path | None,
    calendar_preds: Path,
    pace_preds: Path,
    include_arimax_pace: bool = False,
    debug: bool = False,
    horizons: list[int] | None = None,
    origin_start: str | None = None,
    origin_end: str | None = None,
    origin_stride: int = 1,
    force: bool = False,
) -> None:
    hs = horizons or DEFAULT_ARIMAX_HORIZONS

    if include_arimax_pace:
        if (not force) and calendar_preds.exists() and pace_preds.exists():
            if debug:
                print("[DEBUG] Skip ARIMAX run: both prediction files already exist.", flush=True)
            return
    else:
        if (not force) and calendar_preds.exists():
            if debug:
                print("[DEBUG] Skip ARIMAX run: calendar prediction file already exists.", flush=True)
            return

    raw_v3 = _read_full(DEFAULT_V3_SOURCE)
    ts = _normalize_v3_timeseries(raw_v3)

    eval_keys = _load_reference_eval_keys(run_dir, hs)
    base = _build_eval_rows_from_v3(
        ts, hs, origin_start, origin_end, origin_stride, eval_keys=eval_keys
    )

    key = ["property_id", "origin_date", "stay_date", "exact_horizon"]
    miss = eval_keys.merge(base[key], on=key, how="left", indicator=True)
    miss_n = int((miss["_merge"] == "left_only").sum())
    if miss_n > 0:
        sample = miss.loc[miss["_merge"] == "left_only", key].head(10).to_dict("records")
        raise RuntimeError(
            f"ARIMAX missing {miss_n} eval keys from benchmark universe. Sample: {sample}"
        )

    if debug:
        print(f"[DEBUG] exact eval keys matched: {len(base):,}", flush=True)

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from deeppace_eval.arimax import ArimaxCaseSpec, run_arimax_case  # noqa: WPS433

    if debug:
        print(f"[DEBUG] run_dir={run_dir}", flush=True)
        print(f"[DEBUG] calendar_preds={calendar_preds}", flush=True)
        print(f"[DEBUG] pace_preds={pace_preds}", flush=True)
        print(f"[DEBUG] source_v3={DEFAULT_V3_SOURCE}", flush=True)

    cal = run_arimax_case(
        base,
        spec=ArimaxCaseSpec(name="ARIMAX_CAL", use_calendar=True, use_pace=False),
    )
    calendar_preds.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(calendar_preds, index=False)

    if not calendar_preds.exists() or calendar_preds.stat().st_size == 0:
        raise RuntimeError(f"Failed to write ARIMAX_CAL predictions: {calendar_preds}")

    if debug:
        print(f"[DEBUG] wrote ARIMAX_CAL: {calendar_preds} (rows={len(cal):,})", flush=True)

    if include_arimax_pace:
        pace = run_arimax_case(
            base,
            spec=ArimaxCaseSpec(name="ARIMAX_PACE", use_calendar=False, use_pace=True),
        )
        pace_preds.parent.mkdir(parents=True, exist_ok=True)
        pace.to_csv(pace_preds, index=False)

        if not pace_preds.exists() or pace_preds.stat().st_size == 0:
            raise RuntimeError(f"Failed to write ARIMAX_PACE predictions: {pace_preds}")

        if debug:
            print(f"[DEBUG] wrote ARIMAX_PACE: {pace_preds} (rows={len(pace):,})", flush=True)


def _build_eval_rows_from_v3(
    ts: pd.DataFrame,
    horizons: list[int],
    origin_start: str | None,
    origin_end: str | None,
    origin_stride: int,
    eval_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ts = ts.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["property_id", "date", "y_true"])

    y_lookup = ts.rename(columns={"date": "stay_date"})[["property_id", "stay_date", "y_true"]]

    if eval_keys is not None:
        base = eval_keys.copy()
    else:
        # generate base rows by origins/horizons
        global_min = ts["date"].min()
        global_max = ts["date"].max()
        o_start = pd.to_datetime(origin_start) if origin_start else global_min
        o_end = pd.to_datetime(origin_end) if origin_end else global_max

        rows = []
        for pid, g in ts.groupby("property_id", sort=False):
            dates = pd.to_datetime(g["date"].sort_values().unique())
            origins = [d for d in dates if (d >= o_start and d <= o_end)]
            if origin_stride > 1:
                origins = origins[::origin_stride]
            for origin in origins:
                for h in horizons:
                    stay = pd.Timestamp(origin) + timedelta(days=int(h))
                    rows.append((pid, pd.Timestamp(origin), stay, int(h)))

        base = pd.DataFrame(rows, columns=["property_id", "origin_date", "stay_date", "exact_horizon"])

    out = base.merge(y_lookup, on=["property_id", "stay_date"], how="left")
    out = out.dropna(subset=["y_true"]).copy()
    return out.sort_values(["property_id", "origin_date", "stay_date"], kind="stable")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--calendar-preds", type=Path, default=None)
    p.add_argument("--pace-preds", type=Path, default=None)
    p.add_argument("--input-rows", type=Path, default=None,
                   help="Ignored for ARIMAX generation in this mode.")
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--include-arimax-pace", action="store_true")
    # removed: --source-v3
    p.add_argument(
        "--horizons",
        type=str,
        default="7,10,30,90,180",
        help="Comma-separated ARIMAX horizons (default: 7,10,30,90,180)",
    )
    p.add_argument("--origin-start", type=str, default=None)
    p.add_argument("--origin-end", type=str, default=None)
    p.add_argument("--origin-stride", type=int, default=1)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute ARIMAX even if prediction files exist.")
    args = p.parse_args()

    run_dir = args.run_dir
    report = run_dir / "report.md"
    cal_path = args.calendar_preds or (run_dir / "arimax_calendar_predictions.csv")
    pace_path = args.pace_preds or (run_dir / "arimax_pace_predictions.csv")

    if args.debug:
        print(f"[DEBUG] run_dir={run_dir}")
        print(f"[DEBUG] report={report}")
        print(f"[DEBUG] cal_path={cal_path}")
        print(f"[DEBUG] pace_path={pace_path}")

    run_arimax_if_needed(
        run_dir,
        args.input_rows,
        cal_path,
        pace_path,
        include_arimax_pace=args.include_arimax_pace,
        debug=args.debug,
        horizons=_parse_horizons(args.horizons),
        origin_start=args.origin_start,
        origin_end=args.origin_end,
        origin_stride=args.origin_stride,
        force=args.force,
    )

    if not cal_path.exists():
        raise FileNotFoundError(f"ARIMAX_CAL predictions not found: {cal_path}")

    cal = pd.read_csv(cal_path)
    if cal.empty:
        raise ValueError("ARIMAX_CAL predictions are empty.")
    hz1 = detect_horizon_col(cal, args.horizon_col)
    o1, h1 = compute_metrics(cal, "ARIMAX_CAL", hz1)

    overall_parts = [o1]
    by_h_parts = [h1]

    if args.include_arimax_pace:
        if not pace_path.exists():
            raise FileNotFoundError(f"ARIMAX_PACE predictions not found: {pace_path}")
        pac = pd.read_csv(pace_path)
        if pac.empty:
            raise ValueError("ARIMAX_PACE predictions are empty.")
        hz2 = detect_horizon_col(pac, args.horizon_col)
        o2, h2 = compute_metrics(pac, "ARIMAX_PACE", hz2)
        overall_parts.append(o2)
        by_h_parts.append(h2)

    overall = pd.concat(overall_parts, ignore_index=True)
    by_h = pd.concat(by_h_parts, ignore_index=True).sort_values(["Horizon", "Case"])

    # persist addon metrics for audit
    overall.to_csv(run_dir / "arimax_overall_metrics.csv", index=False)
    by_h.to_csv(run_dir / "arimax_horizon_metrics.csv", index=False)

    section = (
        f"{START}\n"
        "## ARIMAX add-on (without rerunning A/B/C/D)\n\n"
        "### Overall\n\n"
        f"{md_table(overall)}\n\n"
        "### By horizon\n\n"
        f"{md_table(by_h)}\n"
        f"{END}\n"
    )

    base_h = _load_baseline_by_horizon(run_dir)               # from comparisons.csv
    arimax_h = by_h.copy()                                    # your computed ARIMAX by horizon
    merged_h = pd.concat([base_h, arimax_h], ignore_index=True)

    horizon_section = _render_by_h_tables(merged_h)

    text = report.read_text(encoding="utf-8")
    text = _update_main_comparison_table(text, overall)
    text = upsert_section(text, section, START, END)                  # ARIMAX addon block
    text = upsert_section(text, horizon_section, H_START, H_END)      # horizon comparison block
    report.write_text(text, encoding="utf-8")

    verify = report.read_text(encoding="utf-8")
    if START not in verify or END not in verify:
        raise RuntimeError("ARIMAX section was not inserted into report.md")

    print(f"[OK] Updated report: {report}")
    print(f"Wrote {run_dir / 'arimax_overall_metrics.csv'}")
    print(f"Wrote {run_dir / 'arimax_horizon_metrics.csv'}")


def _parse_md_table_block(text: str):
    m = re.search(r"(?ms)^\|.*\|\n^\|[-:| ]+\|\n(?:^\|.*\|\n?)+", text)
    if not m:
        return None
    lines = [ln.rstrip() for ln in m.group(0).strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    align = lines[1]
    rows = lines[2:]
    return m.start(), m.end(), header, align, rows


def _update_main_comparison_table(report_text: str, arimax_overall: pd.DataFrame) -> str:
    parsed = _parse_md_table_block(report_text)
    if parsed is None:
        return report_text

    start, end, cols, align, data_rows = parsed
    row_map = {}

    for ln in data_rows:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != len(cols):
            continue
        row = dict(zip(cols, parts))
        case = row.get("Case")
        if case:
            row_map[case] = row

    for _, r in arimax_overall.iterrows():
        case = str(r["Case"])
        row_map[case] = {
            "Case": case,
            "WMAPE": str(r["WMAPE"]),
            "MAE": str(r["MAE"]),
            "Bias": str(r["Bias"]),
            "MAPE": str(r["MAPE"]),
            "N": str(r["N"]),
        }

    preferred = ["A", "A_bare", "B", "C", "D", "D_shared", "ARIMAX_CAL", "ARIMAX_PACE"]
    ordered = [c for c in preferred if c in row_map] + sorted(set(row_map) - set(preferred))

    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append(align)
    for c in ordered:
        out.append("| " + " | ".join(row_map[c].get(col, "") for col in cols) + " |")
    new_block = "\n".join(out) + "\n"

    return report_text[:start] + new_block + report_text[end:]


def _normalize_eval_keys(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    prop = _pick_col(cols, ["property_id", "property", "hotel_id", "pid"]) or _pick_col_fuzzy(cols, ["property", "hotel"])
    origin = _pick_col(cols, ["origin_date", "origin", "forecast_origin", "as_of_date", "as_of"])
    stay = _pick_col(cols, ["stay_date", "target_date", "date", "arrival_date", "ds"])
    hz = _pick_col(cols, ["exact_horizon", "horizon", "fh", "lead", "days_ahead"])

    if not (prop and origin and (stay or hz)):
        raise ValueError(f"Cannot map eval keys from columns: {cols}")

    out = df.rename(columns={prop: "property_id", origin: "origin_date"}).copy()
    out["origin_date"] = pd.to_datetime(out["origin_date"], errors="coerce")

    if stay:
        out = out.rename(columns={stay: "stay_date"})
        out["stay_date"] = pd.to_datetime(out["stay_date"], errors="coerce")
    else:
        out = out.rename(columns={hz: "exact_horizon"})
        out["exact_horizon"] = pd.to_numeric(out["exact_horizon"], errors="coerce")
        out["stay_date"] = out["origin_date"] + pd.to_timedelta(out["exact_horizon"], unit="D")

    if "exact_horizon" not in out.columns:
        out["exact_horizon"] = (out["stay_date"] - out["origin_date"]).dt.days

    # keep test split only if available
    if "split" in out.columns:
        m = out["split"].astype(str).str.lower().eq("test")
        if m.any():
            out = out.loc[m].copy()

    key = ["property_id", "origin_date", "stay_date", "exact_horizon"]
    out = out.dropna(subset=["property_id", "origin_date", "stay_date"]).copy()
    out = out[key].drop_duplicates().sort_values(key, kind="stable")
    return out


def _load_eval_keys_from_run_dir(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "validation.parquet",
        run_dir / "validation.csv",
        run_dir / "predictions.parquet",
        run_dir / "predictions.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("No validation/predictions file found in run_dir for key matching.")
    return _normalize_eval_keys(_read_full(src))


def _eval_arimax_metrics(df: pd.DataFrame, horizon_col: str) -> pd.DataFrame:
    req = {"y_true", "y_pred", horizon_col}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for metrics computation: {sorted(missing)}")

    d = df.copy()
    d["err"] = d["y_pred"] - d["y_true"]
    d["abs_err"] = d["err"].abs()
    d["ape"] = np.where(d["y_true"] != 0, d["abs_err"] / d["y_true"].abs(), np.nan)

    # horizon-level metrics
    horizon_metrics = (
        d.groupby(horizon_col)
        .agg(
            WMAPE=("abs_err", lambda x: x.sum() / x.index.nunique() if x.index.nunique() > 0 else np.nan),
            MAE=("abs_err", "mean"),
            Bias=("err", "mean"),
            MAPE=("ape", "mean"),
            N=("y_true", "count"),
        )
        .reset_index()
    )

    # overall metrics
    overall = pd.Series(
        {
            "WMAPE": d["abs_err"].sum() / d["y_true"].abs().sum() if d["y_true"].abs().sum() > 0 else np.nan,
            "MAE": d["abs_err"].mean(),
            "Bias": d["err"].mean(),
            "MAPE": d["ape"].mean(),
            "N": d["y_true"].count(),
        }
    )

    return pd.concat([overall.to_frame().T, horizon_metrics], ignore_index=True)


def _debug_arimax_eval(df: pd.DataFrame, label: str) -> None:
    print(f"[DEBUG] {label}: rows={len(df):,}, cols={len(df.columns)}")
    print(f"[DEBUG] columns: {list(df.columns)}")

    for c in ("property_id", "origin_date", "stay_date", "y_true"):
        if c in df.columns:
            na = int(df[c].isna().sum())
            print(f"[DEBUG] {c}: nulls={na:,}")

    if "property_id" in df.columns:
        print(f"[DEBUG] unique properties: {df['property_id'].nunique():,}")
    if "origin_date" in df.columns:
        print(f"[DEBUG] unique origins: {df['origin_date'].nunique():,}")
        print(f"[DEBUG] origin_date range: {df['origin_date'].min()} -> {df['origin_date'].max()}")
    if "stay_date" in df.columns:
        print(f"[DEBUG] stay_date range: {df['stay_date'].min()} -> {df['stay_date'].max()}")

    key_cols = [c for c in ("property_id", "origin_date", "stay_date") if c in df.columns]
    if len(key_cols) == 3:
        dups = int(df.duplicated(key_cols).sum())
        print(f"[DEBUG] duplicate keys ({key_cols}): {dups:,}")


def _arimax_eval_report(df: pd.DataFrame, horizon_col: str, case_name: str) -> str:
    metrics = _eval_arimax_metrics(df, horizon_col)
    overall = metrics.iloc[0]
    by_horizon = metrics.iloc[1:]

    lines = [f"## {case_name} - ARIMAX Evaluation Report\n"]
    lines.append("### Overall Metrics\n")
    lines.append(md_table(pd.DataFrame([overall])))
    lines.append("\n### Metrics by Horizon\n")
    lines.append(md_table(by_horizon))
    return "\n".join(lines)


def _arimax_eval_main(run_dir: Path, cal_path: Path, pace_path: Path, horizons: list[int], debug: bool) -> None:
    cal = pd.read_csv(cal_path)
    if cal.empty:
        raise ValueError("ARIMAX_CAL predictions are empty.")
    hz1 = detect_horizon_col(cal, None)
    o1, h1 = compute_metrics(cal, "ARIMAX_CAL", hz1)

    if debug:
        _debug_arimax_eval(cal, "ARIMAX_CAL predictions")

    overall_parts = [o1]
    by_h_parts = [h1]

    if pace_path and pace_path.exists():
        pac = pd.read_csv(pace_path)
        if pac.empty:
            raise ValueError("ARIMAX_PACE predictions are empty.")
        hz2 = detect_horizon_col(pac, None)
        o2, h2 = compute_metrics(pac, "ARIMAX_PACE", hz2)
        overall_parts.append(o2)
        by_h_parts.append(h2)

        if debug:
            _debug_arimax_eval(pac, "ARIMAX_PACE predictions")

    overall = pd.concat(overall_parts, ignore_index=True)
    by_h = pd.concat(by_h_parts, ignore_index=True).sort_values(["Horizon", "Case"])

    # persist addon metrics for audit
    overall.to_csv(run_dir / "arimax_overall_metrics.csv", index=False)
    by_h.to_csv(run_dir / "arimax_horizon_metrics.csv", index=False)


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None


def _parse_markdown_table(md: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return pd.DataFrame()

    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        vals = [c.strip() for c in ln.strip("|").split("|")]
        if len(vals) == len(header):
            rows.append(vals)

    if not rows:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(rows, columns=header)


def _normalize_horizon_metrics_df(df: pd.DataFrame) -> pd.DataFrame | None:
    d = df.copy()
    d = d.rename(
        columns={
            "horizon": "Horizon",
            "exact_horizon": "Horizon",
            "case": "Case",
            "wmape": "WMAPE",
            "mae": "MAE",
            "bias": "Bias",
            "mape": "MAPE",
            "n": "N",
        }
    )
    need = ["Horizon", "Case", "WMAPE", "MAE", "Bias", "MAPE", "N"]
    if any(c not in d.columns for c in need):
        return None

    out = d[need].copy()
    out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out = out.dropna(subset=["Horizon"]).copy()
    if out.empty:
        return None
    out["Horizon"] = out["Horizon"].astype(int)
    return out


def _is_arimax_case_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.startswith("ARIMAX")


def _discover_baseline_horizon_file(run_dir: Path) -> Path | None:
    patterns = [
        "*horizon*metrics*.csv",
        "*horizon*metrics*.parquet",
        "*horizon*metrics*.feather",
        "*by_horizon*.csv",
        "*by_horizon*.parquet",
        "*by_horizon*.feather",
        "*metrics*.csv",
        "*metrics*.parquet",
        "*metrics*.feather",
    ]
    for pat in patterns:
        for p in sorted(run_dir.rglob(pat)):
            # skip ARIMAX-generated metric artifacts
            if "arimax_" in p.name.lower():
                continue
            try:
                d = _read_full(p)
                norm = _normalize_horizon_metrics_df(d)
                if norm is None:
                    continue
                # require at least one non-ARIMAX case
                if (~_is_arimax_case_series(norm["Case"])).any():
                    return p
            except Exception:
                continue
    return None


def _load_baseline_by_horizon(run_dir: Path) -> pd.DataFrame:
    """
    Load baseline by-horizon metrics for A/B/C/D...
    Fallback:
      1) report.md HORIZON_METRICS block
      2) horizon_metrics.csv
      3) discovered metrics file (excluding arimax_*)
      4) compute from predictions/validation
    """
    report = run_dir / "report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        s_tag = "<!-- HORIZON_METRICS:START -->"
        e_tag = "<!-- HORIZON_METRICS:END -->"
        if s_tag in text and e_tag in text:
            s = text.index(s_tag) + len(s_tag)
            e = text.index(e_tag)
            block = text[s:e].strip()
            df = _parse_markdown_table(block)
            norm = _normalize_horizon_metrics_df(df)
            if norm is not None:
                norm = norm.loc[~_is_arimax_case_series(norm["Case"])].copy()
                if not norm.empty:
                    return norm

    csv_fallback = run_dir / "horizon_metrics.csv"
    if csv_fallback.exists():
        norm = _normalize_horizon_metrics_df(pd.read_csv(csv_fallback))
        if norm is not None:
            norm = norm.loc[~_is_arimax_case_series(norm["Case"])].copy()
            if not norm.empty:
                return norm

    discovered = _discover_baseline_horizon_file(run_dir)
    if discovered is not None:
        norm = _normalize_horizon_metrics_df(_read_full(discovered))
        if norm is not None:
            norm = norm.loc[~_is_arimax_case_series(norm["Case"])].copy()
            if not norm.empty:
                return norm

    # final fallback: derive A/B/C/D by horizon from predictions
    return _compute_baseline_by_horizon_from_predictions(run_dir)


def _render_by_h_tables(df: pd.DataFrame) -> str:
    df = df.copy()
    if "Horizon" not in df.columns:
        return ""

    needed = ["Horizon", "Case", "WMAPE", "MAE", "Bias", "MAPE", "N"]
    df = df[needed].copy()

    order = ["A", "A_bare", "B", "C", "D", "D_shared", "ARIMAX_CAL", "ARIMAX_PACE"]
    df["Case"] = pd.Categorical(df["Case"], categories=order, ordered=True)

    horizons = sorted(df["Horizon"].dropna().unique().tolist())
    parts = [H_START, "## Comparison by horizon", ""]

    # split into chunks of 6 horizons per section
    for i in range(0, len(horizons), 6):
        chunk = horizons[i : i + 6]
        if not chunk:
            continue

        parts.append(f"### Horizons {chunk[0]} to {chunk[-1]}")
        parts.append("")

        for h in chunk:
            x = df[df["Horizon"] == h].sort_values("Case")
            parts.append(f"#### Horizon = {h}")
            parts.append("")

            tbl = x[["Case", "WMAPE", "MAE", "Bias", "MAPE", "N"]].copy()
            # avoid fillna on categorical
            tbl["Case"] = tbl["Case"].astype("string")
            for c in ["WMAPE", "MAE", "Bias", "MAPE", "N"]:
                tbl[c] = tbl[c].where(pd.notna(tbl[c]), "")

            parts.append(md_table(tbl))
            parts.append("")

    parts.append(H_END)
    parts.append("")
    return "\n".join(parts)


def _load_pairwise_from_comparisons(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "comparisons.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing comparisons.csv: {p}")

    d = pd.read_csv(p)
    req = ["candidate", "reference", "segment", "delta_wmape", "delta_mae"]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise ValueError(f"comparisons.csv missing required columns: {miss}")

    seg = d["segment"].astype(str)
    hz = seg.str.extract(r"horizon[_\- ]?(\d+)", expand=False)
    d = d[hz.notna()].copy()
    d["Horizon"] = hz[hz.notna()].astype(int)

    # keep useful columns if present
    keep = [
        "Horizon",
        "candidate",
        "reference",
        "delta_wmape",
        "delta_mae",
        "n_origins",
        "block_origins",
        "ci_status",
        "property_win_rate",
        "horizon_win_rate",
        "row_win_rate",
        "row_tie_rate",
        "wmape_low",
        "wmape_high",
        "mae_low",
        "mae_high",
    ]
    keep = [c for c in keep if c in d.columns]
    return d[keep].sort_values(["Horizon", "candidate", "reference"], kind="stable")


def _render_pairwise_horizon_tables(df: pd.DataFrame) -> str:
    parts = [H_START, "## Comparison by horizon (from comparisons.csv)", ""]
    for h in sorted(df["Horizon"].unique()):
        x = df[df["Horizon"] == h].copy()
        parts.append(f"### Horizon = {h}")
        parts.append("")
        parts.append(md_table(x))
        parts.append("")
    parts.append(H_END)
    parts.append("")
    return "\n".join(parts)


def _compute_baseline_by_horizon_from_predictions(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "predictions.parquet",
        run_dir / "predictions.csv",
        run_dir / "validation.parquet",
        run_dir / "validation.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "No predictions/validation file found in run_dir to compute baseline horizon metrics."
        )

    d = _read_full(src).copy()
    cols = list(d.columns)

    c_case = _pick_col(cols, ["case", "model_case", "candidate", "method"])
    c_true = _pick_col(cols, ["y_true", "actual", "target", "rooms_sold"])
    c_pred = _pick_col(cols, ["y_pred", "pred", "prediction", "yhat"])
    c_h = _pick_col(cols, ["exact_horizon", "horizon", "fh", "lead", "days_ahead"])
    c_o = _pick_col(cols, ["origin_date", "origin", "as_of_date", "as_of"])
    c_s = _pick_col(cols, ["stay_date", "target_date", "date", "arrival_date", "ds"])

    if not (c_case and c_true and c_pred):
        raise ValueError(f"Cannot map Case/y_true/y_pred from {src}. Columns: {cols}")

    d = d.rename(columns={c_case: "Case", c_true: "y_true", c_pred: "y_pred"})
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce")

    if c_h:
        d = d.rename(columns={c_h: "Horizon"})
        d["Horizon"] = pd.to_numeric(d["Horizon"], errors="coerce")
    else:
        if not (c_o and c_s):
            raise ValueError("Need horizon column or origin_date+stay_date to derive horizon.")
        d["origin_date"] = pd.to_datetime(d[c_o], errors="coerce")
        d["stay_date"] = pd.to_datetime(d[c_s], errors="coerce")
        d["Horizon"] = (d["stay_date"] - d["origin_date"]).dt.days

    if "split" in d.columns:
        m = d["split"].astype(str).str.lower().eq("test")
        if m.any():
            d = d.loc[m].copy()

    d = d.dropna(subset=["Case", "Horizon", "y_true", "y_pred"]).copy()
    d["Horizon"] = d["Horizon"].astype(int)

    # baseline only (exclude ARIMAX; it is added separately)
    d = d.loc[~d["Case"].astype(str).str.upper().str.startswith("ARIMAX")].copy()
    if d.empty:
        raise ValueError("No non-ARIMAX baseline rows found in predictions/validation file.")

    parts: list[pd.DataFrame] = []
    for case_name, g in d.groupby("Case", sort=True):
        _, by_h = compute_metrics(g, str(case_name), "Horizon")
        parts.append(by_h)

    out = pd.concat(parts, ignore_index=True)
    out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").astype(int)
    return out.sort_values(["Horizon", "Case"], kind="stable")


def _load_reference_eval_keys(run_dir: Path, horizons: list[int]) -> pd.DataFrame:
    candidates = [
        run_dir / "predictions.parquet",
        run_dir / "predictions.csv",
        run_dir / "validation.parquet",
        run_dir / "validation.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("No predictions/validation file found to derive eval keys.")

    d = _read_full(src).copy()
    cols = list(d.columns)

    c_prop = _pick_col(cols, ["property_id", "property", "hotel_id", "pid"])
    c_org = _pick_col(cols, ["origin_date", "origin", "as_of_date", "as_of"])
    c_stay = _pick_col(cols, ["stay_date", "target_date", "date", "arrival_date", "ds"])
    c_h = _pick_col(cols, ["exact_horizon", "horizon", "fh", "lead", "days_ahead"])

    if not (c_prop and c_org and (c_stay or c_h)):
        raise ValueError(f"Cannot map eval keys from {src}. Columns: {cols}")

    out = d.rename(columns={c_prop: "property_id", c_org: "origin_date"}).copy()
    out["origin_date"] = pd.to_datetime(out["origin_date"], errors="coerce")

    if c_stay:
        out = out.rename(columns={c_stay: "stay_date"})
        out["stay_date"] = pd.to_datetime(out["stay_date"], errors="coerce")
    else:
        out = out.rename(columns={c_h: "exact_horizon"})
        out["exact_horizon"] = pd.to_numeric(out["exact_horizon"], errors="coerce")
        out["stay_date"] = out["origin_date"] + pd.to_timedelta(out["exact_horizon"], unit="D")

    if "exact_horizon" not in out.columns:
        out["exact_horizon"] = (out["stay_date"] - out["origin_date"]).dt.days

    if "split" in out.columns:
        m = out["split"].astype(str).str.lower().eq("test")
        if m.any():
            out = out.loc[m].copy()

    out["exact_horizon"] = pd.to_numeric(out["exact_horizon"], errors="coerce")
    out = out.dropna(subset=["property_id", "origin_date", "stay_date", "exact_horizon"]).copy()
    out["exact_horizon"] = out["exact_horizon"].astype(int)
    out = out[out["exact_horizon"].isin(horizons)].copy()

    key = ["property_id", "origin_date", "stay_date", "exact_horizon"]
    return out[key].drop_duplicates().sort_values(key, kind="stable")


if __name__ == "__main__":
    main()