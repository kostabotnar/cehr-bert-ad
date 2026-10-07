#!/usr/bin/env python
"""Step 18: Profile the raw OMOP data split into AD cases and controls.

Reads the OMOP tables directly (nothing from step 6 or later) and describes AD cases and controls
under three index views: ``ad`` (first AD diagnosis), ``control_last_visit`` (last visit) and
``control_matched_index`` (a visit date of the control chosen so its age band and calendar year
distribution follows the cases, see ``pipeline/omop_profile.py``). The goal is to choose cohort
rules (date floor, case definition, control exclusions, index rule, minimum history) before
training. Nothing is applied to any cohort file.

Outputs (under build/omop_profile/ by default)
  profile.json         all sections, by view
  table1.csv / .md     one row per variable level, SMD of each control view versus AD
  exclusion_flow.csv   sequential candidate exclusions (reported only)
  figures/*.png        case/control overlays (300 dpi)
  reports/18_profile_omop/{status.json,report.md}

Usage
  python 18_profile_omop.py
  python 18_profile_omop.py --person-sample-fraction 0.02 --skip-sections measurement,duplicates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import omop_profile as op  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.omop_validation import table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

STEP = "18_profile_omop"
SECTIONS = (
    "data_quality",
    "demographics",
    "index_date",
    "history",
    "utilization",
    "clinical_content",
    "outcome_validity",
    "control_contamination",
    "site_source",
)
# Extra skip tokens for the most expensive parts.
EXTRA_SKIPS = ("measurement", "duplicates")
SMD_WARN = 0.1
CONTAMINATION_WARN = 0.05
PERSON_COLS = [
    "person_id",
    "gender_concept_id",
    "year_of_birth",
    "month_of_birth",
    "day_of_birth",
    "race_concept_id",
    "ethnicity_concept_id",
    "location_id",
    "care_site_id",
]
VISIT_COLS = [
    "visit_occurrence_id",
    "person_id",
    "visit_start_date",
    "visit_concept_id",
    "visit_type_concept_id",
    "provider_id",
    "care_site_id",
]
VIEW_COLORS = {op.VIEW_AD: "#D55E00", op.VIEW_LAST: "#0072B2", op.VIEW_MATCHED: "#009E73"}
TABLE1_CONTINUOUS = ["age", "birth_year", "index_year", "history_years", "visits_per_year_w1"]
TABLE1_CATEGORICAL = ["age_band_label", "sex", "race", "ethnicity"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")


def _read_release_date(data_dir: Path):
    glob = table_glob(data_dir, "cdm_source")
    if glob is None:
        return None
    try:
        df = pl.read_parquet(glob, columns=["source_release_date"])
        return df["source_release_date"].cast(pl.Date)[0] if df.height else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _overlay_hist(feats: dict[str, pl.DataFrame], col: str, bins, xlabel: str, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for view, df in feats.items():
        vals = df[col].drop_nulls().to_numpy()
        if vals.size:
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=VIEW_COLORS[view],
                label=f"{view} (n={vals.size})",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, out)


def plot_age_hist(feats, out: Path) -> Path:
    return _overlay_hist(feats, "age", np.arange(0, 111, 2), "age at index (years)", "Age at index", out)


def plot_index_year_hist(feats, out: Path) -> Path:
    years = [y for df in feats.values() for y in df["index_year"].drop_nulls().to_list()]
    lo, hi = (min(years), max(years)) if years else (1990, 2026)
    return _overlay_hist(feats, "index_year", np.arange(lo, hi + 2) - 0.5, "index year", "Index calendar year", out)


def plot_year_age_heatmaps(feats, out_dir: Path) -> list[Path]:
    outs = []
    for view, df in feats.items():
        d = df.drop_nulls(["index_year", "age"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if d.height:
            h = ax.hist2d(
                d["index_year"].to_numpy(),
                d["age"].to_numpy(),
                bins=[np.arange(d["index_year"].min(), d["index_year"].max() + 2) - 0.5, np.arange(0, 111, 5)],
                cmap="viridis",
            )
            fig.colorbar(h[3], ax=ax, label="persons")
        ax.set_xlabel("index year")
        ax.set_ylabel("age at index (years)")
        ax.set_title(f"Index year by age: {view}")
        outs.append(_save(fig, out_dir / f"year_by_age_heatmap_{view}.png"))
    return outs


def plot_history_cdf(feats, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for view, df in feats.items():
        v = np.sort(df["history_years"].drop_nulls().to_numpy())
        if v.size:
            ax.plot(
                v,
                np.arange(1, v.size + 1) / v.size,
                color=VIEW_COLORS[view],
                linewidth=1.8,
                label=f"{view} (n={v.size})",
            )
    ax.set_xlabel("years of history before index")
    ax.set_ylabel("cumulative share of persons")
    ax.set_title("History length before index")
    ax.legend(frameon=False)
    return _save(fig, out)


def plot_monthly_density(monthly_per_patient: dict[str, dict[str, float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for view, months in monthly_per_patient.items():
        x = np.arange(36)
        y = [months.get(str(m), 0.0) for m in x]
        ax.plot(-(x + 0.5) * 30 / 30.4375, y, color=VIEW_COLORS[view], linewidth=1.8, label=view)
    ax.set_xlabel("months before index (30 day blocks)")
    ax.set_ylabel("visits per patient per block")
    ax.set_title("Visit density before index")
    ax.legend(frameon=False)
    return _save(fig, out)


def plot_visit_type_mix(shares: dict[str, dict[str, float | None]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    views = list(shares)
    width = 0.8 / max(len(views), 1)
    x = np.arange(len(op.VISIT_TYPE_ORDER))
    for i, view in enumerate(views):
        y = [shares[view].get(b) or 0.0 for b in op.VISIT_TYPE_ORDER]
        ax.bar(x + i * width, y, width, color=VIEW_COLORS[view], label=view)
    ax.set_xticks(x + width * (len(views) - 1) / 2)
    ax.set_xticklabels(op.VISIT_TYPE_ORDER)
    ax.set_ylabel("share of visits before index")
    ax.set_title("Visit type mix before index")
    ax.legend(frameon=False)
    return _save(fig, out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile raw OMOP data for AD cases versus controls")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--out-dir", default=str(paths.OMOP_PROFILE_DIR))
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--concepts", default=str(paths.CONFIG_DIR / "omop_profile_concepts.yaml"))
    parser.add_argument("--date-floor", default=op.DEFAULT_DATE_FLOOR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--person-sample-fraction", type=float, default=1.0)
    parser.add_argument(
        "--skip-sections",
        default="",
        help=f"comma list of {', '.join(SECTIONS + EXTRA_SKIPS)}",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir) if args.report_dir else paths.report_dir_for(STEP)
    skip = {s.strip() for s in args.skip_sections.split(",") if s.strip()}
    unknown = skip - set(SECTIONS) - set(EXTRA_SKIPS)
    if unknown:
        parser.error(f"unknown --skip-sections: {sorted(unknown)}")

    report = StepReport(STEP, title="Step 18 - Raw OMOP profile of AD cases versus controls")
    report.add_metric("person_sample_fraction", args.person_sample_fraction)
    report.add_metric("seed", args.seed)

    def finish() -> int:
        report.write(report_dir)
        report.print_summary()
        print(f"\nReport written to {report_dir}")
        return report.exit_code()

    required = ["person", "visit_occurrence", "condition_occurrence"]
    missing = [t for t in required if table_glob(data_dir, t) is None]
    report.add_check(
        "input tables found", not missing, detail=f"missing: {missing}" if missing else f"{required} under {data_dir}"
    )
    if missing:
        return finish()

    cfg = op.load_concept_config(args.concepts)
    sample = op.sample_expr(args.person_sample_fraction, args.seed)
    release_date = _read_release_date(data_dir)
    report.add_metric("source_release_date", release_date)

    # --- persons, visits, AD rows ------------------------------------------------------------
    log("loading persons")
    persons0 = op.load_persons(
        op.scan_table(data_dir, "person", PERSON_COLS, sample),
        op.scan_table(data_dir, "death", ["person_id", "death_date"], sample),
        op.scan_table(
            data_dir,
            "observation_period",
            ["person_id", "observation_period_start_date", "observation_period_end_date"],
            sample,
        ),
    )
    log(f"{persons0.height} persons; loading visits")
    visits = op.scan_table(data_dir, "visit_occurrence", VISIT_COLS, sample).collect(engine="streaming")
    vsum = op.visit_summary(visits)
    log(f"{visits.height} visits; scanning conditions")

    cond_items = op.regex_items(cfg["conditions"])
    proc_items = op.regex_items(cfg["procedures"])
    meas_items = op.measurement_items(cfg["measurements"])
    cond_lf = op.scan_table(
        data_dir,
        "condition_occurrence",
        ["person_id", "condition_start_date", "condition_source_value", "visit_occurrence_id"],
        sample,
    ).rename({"condition_start_date": "date", "condition_source_value": "source"})
    union = op.AD_SOURCE_REGEX + ("|" + op.union_regex(cond_items) if cond_items else "")
    cond_rows = cond_lf.filter(pl.col("source").str.contains(union)).collect(engine="streaming")
    ad_rows = cond_rows.filter(pl.col("source").str.contains(op.AD_SOURCE_REGEX)).join(
        persons0.select("person_id"), on="person_id", how="semi"
    )
    ad_first = op.ad_first_dates(ad_rows)
    cond_events = op.condition_item_events(cond_rows, cond_items, "source")
    del cond_rows

    persons = op.assign_cohort(persons0, ad_first)
    n_ad = int((persons["cohort"] == op.COHORT_AD).sum())
    n_ctl = persons.height - n_ad
    report.add_metric("n_persons", persons.height)
    report.add_metric("n_ad", n_ad)
    report.add_metric("n_control", n_ctl)
    report.add_check("both cohorts non-empty", n_ad > 0 and n_ctl > 0, detail=f"ad={n_ad}, control={n_ctl}")
    if n_ad == 0 or n_ctl == 0:
        return finish()

    # --- index dates -------------------------------------------------------------------------
    log("building index views")
    control_ids = persons.filter(pl.col("cohort") == op.COHORT_CONTROL).select("person_id")
    control_dates = visits.join(control_ids, on="person_id", how="semi").select("person_id", "visit_start_date")
    index_table, index_info = op.build_index_table(persons, vsum, control_dates, args.seed)
    del control_dates
    for k, v in index_info.items():
        report.add_metric(k, v)

    feats: dict[str, pl.DataFrame] = {}
    monthly: dict[str, pl.DataFrame] = {}
    visits_core = visits.select("person_id", "visit_start_date", "visit_concept_id", "provider_id", "care_site_id")
    for view in op.VIEWS:
        idx = index_table.filter(pl.col("view") == view).select("person_id", "index_date")
        log(f"visit features for {view} ({idx.height} persons)")
        vfeat, mdf = op.visit_features(visits_core, idx)
        feats[view] = op.view_features(view, persons, idx, vsum, vfeat)
        monthly[view] = mdf
    del visits_core

    # --- clinical events ---------------------------------------------------------------------
    drug_events = pl.DataFrame(schema=cond_events.schema)
    drug_items: list[str] = []
    unresolved: list[str] = []
    drugs_lf = op.scan_table(
        data_dir, "drug_exposure", ["person_id", "drug_concept_id", "drug_exposure_start_date"], sample
    )
    concept_lf = op.scan_table(
        data_dir, "concept", ["concept_id", "concept_name", "vocabulary_id", "concept_class_id", "standard_concept"]
    )
    ancestor_lf = op.scan_table(data_dir, "concept_ancestor", ["ancestor_concept_id", "descendant_concept_id"])
    if drugs_lf is not None and concept_lf is not None and ancestor_lf is not None and cfg["drugs"]:
        log("resolving drug concepts and scanning drug_exposure")
        mapping, unresolved = op.resolve_drug_concepts(concept_lf, ancestor_lf, cfg["drugs"])
        drug_items = [op.item_id(g, n.lower()) for g, ns in cfg["drugs"].items() for n in ns]
        drug_events = (
            drugs_lf.join(mapping.lazy(), on="drug_concept_id")
            .select("person_id", pl.col("drug_exposure_start_date").alias("date"), "group", "item")
            .collect(engine="streaming")
        )
        report.add_metric("drug_names_unresolved", ",".join(unresolved))
    else:
        report.note("drug_exposure, concept or concept_ancestor missing: drug groups skipped")

    proc_events = pl.DataFrame(schema=cond_events.schema)
    want_content = "clinical_content" not in skip
    proc_lf = op.scan_table(
        data_dir, "procedure_occurrence", ["person_id", "procedure_date", "procedure_source_value"], sample
    )
    if want_content and proc_lf is not None and proc_items:
        log("scanning procedures")
        rows = (
            proc_lf.rename({"procedure_date": "date", "procedure_source_value": "source"})
            .filter(pl.col("source").str.contains(op.union_regex(proc_items)))
            .collect(engine="streaming")
        )
        proc_events = op.condition_item_events(rows, proc_items, "source")
    meas_events = pl.DataFrame(schema=cond_events.schema)
    meas_lf = op.scan_table(
        data_dir, "measurement", ["person_id", "measurement_date", "measurement_source_value"], sample
    )
    if want_content and "measurement" not in skip and meas_lf is not None and meas_items:
        log("scanning measurements")
        codes = sorted({c for _, cs in meas_items.values() for c in cs})
        rows = (
            meas_lf.rename({"measurement_date": "date", "measurement_source_value": "source"})
            .filter(pl.col("source").is_in(codes))
            .collect(engine="streaming")
        )
        meas_events = op.exact_item_events(rows, meas_items, "source")
    cond_item_ids = list(cond_items)
    proc_item_ids = list(proc_items)
    meas_item_ids = list(meas_items) if "measurement" not in skip else []
    all_item_ids = cond_item_ids + drug_items + proc_item_ids + meas_item_ids
    events = pl.concat([cond_events, drug_events, proc_events, meas_events])
    flow_events = pl.concat([cond_events, drug_events])

    # --- exclusion flow (always) -------------------------------------------------------------
    contaminated = op.contaminated_controls(flow_events, control_ids["person_id"])
    flow = op.exclusion_flow(persons, vsum, contaminated, args.date_floor)

    profile: dict = {
        "meta": {
            "data_dir": str(data_dir),
            "person_sample_fraction": args.person_sample_fraction,
            "seed": args.seed,
            "date_floor": args.date_floor,
            "source_release_date": release_date,
            "skipped": sorted(skip),
            "n_persons": persons.height,
            "n_ad": n_ad,
            "n_control": n_ctl,
            "index": index_info,
            "unresolved_drug_names": unresolved,
            "windows": "half-open [index - k, index); monthly blocks are 30 days",
        },
        "exclusion_flow": flow.to_dicts(),
    }

    # --- sections ----------------------------------------------------------------------------
    if "data_quality" not in skip:
        log("data quality")
        dq: dict = {"person_fields": op.person_field_quality(persons), "tables": {}}
        for table, (date_col, concept_col) in op.QUALITY_TABLES.items():
            if table == "measurement" and "measurement" in skip:
                continue
            if table == "visit_occurrence":
                lf = visits.lazy().select("person_id", pl.col(date_col), pl.col(concept_col))
            else:
                lf = op.scan_table(data_dir, table, ["person_id", date_col, concept_col], sample)
            if lf is None:
                continue
            log(f"  quality of {table}")
            dq["tables"][table] = op.table_quality(
                lf, persons, date_col, concept_col, release_date, args.date_floor, duplicates="duplicates" not in skip
            )
        profile["data_quality"] = dq

    demo_feats = feats
    if "demographics" not in skip:
        demo = op.demographics_section(demo_feats)
        smd = op.smd_section(
            demo_feats,
            ["age", "birth_year", "index_year", "history_years"],
            ["age_band_label", "sex", "race", "ethnicity"],
        )
        profile["demographics"] = {
            "by_view": demo,
            "smd_vs_ad": smd,
            "age_band_ad_prevalence": op.age_band_prevalence(feats[op.VIEW_AD], feats[op.VIEW_LAST]),
        }
    if "index_date" not in skip:
        profile["index_date"] = op.index_date_section(feats)
    if "history" not in skip:
        profile["history"] = op.history_section(feats)
    if "utilization" not in skip:
        profile["utilization"] = op.utilization_section(feats, monthly)

    binary: dict[str, dict[str, tuple[int, int]]] = {}
    if want_content:
        log("clinical content")
        content: dict = {"all_history": {}, "last_730_days": {}, "ad_drug_first_exposure": {}}
        for view in op.VIEWS:
            idx = feats[view].select("person_id", "index_date")
            content["all_history"][view] = op.prevalence(events, idx, None, all_item_ids)
            content["last_730_days"][view] = op.prevalence(events, idx, 730, all_item_ids)
            content["ad_drug_first_exposure"][view] = op.first_exposure_days(events, idx, "ad_drugs")
        profile["clinical_content"] = content
        for item in all_item_ids:
            binary[f"{item} (before index)"] = {
                v: (content["all_history"][v]["items"][item]["n"], content["all_history"][v]["n"]) for v in op.VIEWS
            }

    if "outcome_validity" not in skip:
        proxy = events.filter(pl.col("group") == "dementia_proxies")
        profile["outcome_validity"] = op.outcome_validity(
            ad_rows, ad_first.join(persons0.select("person_id"), on="person_id", how="semi"), vsum, visits, proxy
        )
    if "control_contamination" not in skip:
        profile["control_contamination"] = op.contamination_section(
            feats, events, [i for i in all_item_ids if op.item_group(i) in ("dementia_proxies", "ad_drugs")]
        )
    if "site_source" not in skip:
        cond_types = None
        ct_lf = op.scan_table(data_dir, "condition_occurrence", ["person_id", "condition_type_concept_id"], sample)
        if ct_lf is not None:
            cond_types = (
                ct_lf.join(persons.lazy().select("person_id", "cohort"), on="person_id")
                .group_by("cohort", "condition_type_concept_id")
                .agg(pl.len().alias("n"))
                .collect(engine="streaming")
            )
        profile["site_source"] = op.site_source_section(persons, visits, cond_types)

    # --- outputs -----------------------------------------------------------------------------
    log("writing outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    table1 = op.build_table1(feats, TABLE1_CONTINUOUS, TABLE1_CATEGORICAL, binary)
    t1_csv = out_dir / "table1.csv"
    pl.DataFrame(table1, infer_schema_length=None).write_csv(t1_csv)
    t1_md = out_dir / "table1.md"
    t1_md.write_text(op.table1_markdown(table1), encoding="utf-8")
    flow_csv = out_dir / "exclusion_flow.csv"
    flow.write_csv(flow_csv)
    profile_json = out_dir / "profile.json"
    profile_json.write_text(json.dumps(profile, indent=2, default=_json_default), encoding="utf-8")

    fig_dir = out_dir / "figures"
    figs = [
        plot_age_hist(feats, fig_dir / "age_at_index_hist.png"),
        plot_index_year_hist(feats, fig_dir / "index_year_hist.png"),
        *plot_year_age_heatmaps(feats, fig_dir),
        plot_history_cdf(feats, fig_dir / "history_cdf.png"),
    ]
    if "utilization" in profile:
        figs.append(
            plot_monthly_density(
                {v: profile["utilization"][v]["monthly_visits_per_patient"] for v in op.VIEWS},
                fig_dir / "monthly_visit_density.png",
            )
        )
        figs.append(
            plot_visit_type_mix(
                {v: profile["utilization"][v]["visit_type_share"] for v in op.VIEWS}, fig_dir / "visit_type_mix.png"
            )
        )
    written = [t1_csv, t1_md, flow_csv, profile_json, *figs]
    for p in written:
        report.add_artifact(p)
    report.add_check(
        "outputs written", all(Path(p).exists() for p in written), detail=f"{len(written)} files in {out_dir}"
    )

    # --- checks and headline metrics ---------------------------------------------------------
    smd_all = profile.get("demographics", {}).get("smd_vs_ad", {})
    for view in op.CONTROL_VIEWS:
        s = smd_all.get(view, {})
        sex = (s.get("sex") or {}).get("female")
        for name, val in (("age", s.get("age")), ("sex (female)", sex), ("index year", s.get("index_year"))):
            if val is not None:
                report.warn(f"SMD {name} {view} vs AD <= {SMD_WARN}", abs(val) <= SMD_WARN, detail=f"smd={val:.3f}")
        if s.get("age") is not None:
            report.add_metric(f"smd_age_{view}", round(s["age"], 4))
        if s.get("index_year") is not None:
            report.add_metric(f"smd_index_year_{view}", round(s["index_year"], 4))
    cont = profile.get("control_contamination", {}).get(op.VIEW_LAST)
    if cont and cont["share_any_proxy_or_ad_drug"] is not None:
        c = cont["share_any_proxy_or_ad_drug"]
        report.add_metric("control_contamination_share", round(c, 4))
        report.warn(
            f"control contamination <= {CONTAMINATION_WARN:.0%}", c <= CONTAMINATION_WARN, detail=f"share={c:.3f}"
        )
    final = flow.group_by("cohort", maintain_order=True).agg(pl.col("n_remaining").last())
    for r in final.iter_rows(named=True):
        report.add_metric(f"flow_remaining_{r['cohort']}", r["n_remaining"])
    report.note("Exclusion flow is sequential and reported only, no exclusion is applied to any cohort file.")
    report.add_metric("wall_seconds", round(time.time() - t0))
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
