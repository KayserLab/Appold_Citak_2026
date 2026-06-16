import concurrent.futures as cf
import pathlib as pl
import numpy as np
import pandas as pd
import validation_common as vc



def get_parameter_columns():
    return ["regrowth_t0_h", "regrowth_slope_um_per_h", "fitted_diffusion_sensitive",
            "fitted_uptake_rate", "fitted_diffusion_nutrients", "fitted_start_point",
            "fitted_mutation_rate"]


def get_metric_columns():
    return ["test_area_mae", "test_area_rmse", "test_area_nrmse", "test_area_r2",
            "test_regrowth_time_mae_h", "test_regrowth_time_rmse_h", "test_regrowth_time_r2",
            "test_abs_error", "test_squared_error", "test_relative_abs_error",
            "test_standardized_residual"]


def get_parameter_row(parameter_summary, parameter_name):
    row = parameter_summary.loc[parameter_summary["parameter"] == parameter_name]
    if row.empty:
        raise ValueError(f"Parameter {parameter_name} not found in parameter summary.")
    return row.iloc[0].to_dict()


def make_headline_row(category, headline_metric, mean_value, median_value, std_value, unit, direction):
    return {"category": category,
            "headline_metric": headline_metric,
            "mean_across_folds": mean_value,
            "median_across_folds": median_value,
            "std_across_folds": std_value,
            "unit": unit,
            "direction": direction}


def build_headline_summary(metric_summary, parameter_summary):
    start_point_row = get_parameter_row(parameter_summary, "fitted_start_point")
    mutation_rate_row = get_parameter_row(parameter_summary, "fitted_mutation_rate")

    rows = [make_headline_row("Held-out no-treatment", "Area NRMSE", metric_summary["test_area_nrmse"]["mean"], metric_summary["test_area_nrmse"]["median"], metric_summary["test_area_nrmse"]["std"], "relative error", "lower is better"), make_headline_row("Held-out no-treatment", "Area R2", metric_summary["test_area_r2"]["mean"], metric_summary["test_area_r2"]["median"], metric_summary["test_area_r2"]["std"], "unitless", "higher is better"), make_headline_row("Held-out continuous therapy", "Regrowth RMSE", metric_summary["test_regrowth_time_rmse_h"]["mean"], metric_summary["test_regrowth_time_rmse_h"]["median"], metric_summary["test_regrowth_time_rmse_h"]["std"], "hours", "lower is better"), make_headline_row("Held-out continuous therapy", "Mutation relative absolute error", metric_summary["test_relative_abs_error"]["mean"], metric_summary["test_relative_abs_error"]["median"], metric_summary["test_relative_abs_error"]["std"], "relative error", "lower is better"), make_headline_row("Parameter stability", "Start point LOO SD", start_point_row["loo_std"], start_point_row["loo_median"], np.nan, "simulation steps", "lower means more stable"), make_headline_row("Parameter stability", "Mutation rate LOO SD", mutation_rate_row["loo_std"], mutation_rate_row["loo_median"], np.nan, "rate units", "lower means more stable")]
    return pd.DataFrame(rows)


def build_headline_summary_text(headline_summary):
    lines = ["End-to-End Validation Headline Summary", "====================================="]
    for _, row in headline_summary.iterrows():
        mean_value = row["mean_across_folds"]
        median_value = row["median_across_folds"]
        std_value = row["std_across_folds"]

        if pd.notna(mean_value):
            line = f"{row['category']} | {row['headline_metric']}: mean={mean_value:.4g}"
        else:
            line = f"{row['category']} | {row['headline_metric']}: mean=nan"

        if pd.notna(median_value):
            line += f", median={median_value:.4g}"
        else:
            line += ", median=nan"
        if pd.notna(std_value):
            line += f", sd={std_value:.4g}"

        line += f" [{row['unit']}]"
        line += f" ({row['direction']})"
        lines.append(line)

    lines.append("")
    lines.append("Use fold_results.csv for the full per-fold breakdown.")
    return "\n".join(lines) + "\n"


def area_observations_from_entries(entries):
    return [np.asarray(entry["area"], dtype=float) for entry in entries]


def clone_count_array(entries):
    return np.asarray([float(entry["clone_count"]) for entry in entries], dtype=float)


def get_entry_by_label(dataset, label):
    return next(entry for entry in dataset if str(entry["label"]) == str(label))


def run_end_to_end_fold(fold_task):
    train_no_treatment = [entry for entry in fold_task["no_treatment_data"]
                          if str(entry["label"]) != fold_task["held_out_no_treatment_label"]]
    train_continuous = [entry for entry in fold_task["continuous_therapy_data"]
                        if str(entry["label"]) != fold_task["held_out_continuous_well"]]
    train_continuous_wells = [str(entry["label"]) for entry in train_continuous]

    regrowth_fit = vc.fit_regrowth_calibration_from_continuous_therapy(wells=train_continuous_wells, min_duration_frames=fold_task["min_duration_frames"], min_distance_um=fold_task["min_distance_um"])

    dispersion_fit = vc.fit_dispersion_dataset(area_observations=area_observations_from_entries(train_no_treatment), initial_guess=fold_task["dispersion_guess"], params_yaml=fold_task["params_yaml"], regrowth_calibration=regrowth_fit, maxiter=fold_task["maxiter_dispersion"])

    fold_start_point = int(dispersion_fit.metadata["start_point"])
    fold_mutation_simulation_overrides = vc.build_dispersion_parameter_overrides(dispersion_fit.result.x)

    mutation_rate_fit = vc.fit_mutation_rate_dataset(clone_counts=clone_count_array(train_continuous), initial_guess=fold_task["mutation_rate_guess"], params_yaml=fold_task["params_yaml"], maxiter=fold_task["maxiter_mutation_rate"], sim_replicas=fold_task["fit_sim_replicas"], start_point_override=fold_start_point, simulation_param_overrides=fold_mutation_simulation_overrides)

    held_out_no_treatment = get_entry_by_label(fold_task["no_treatment_data"], fold_task["held_out_no_treatment_label"])
    held_out_continuous = get_entry_by_label(fold_task["continuous_therapy_data"], fold_task["held_out_continuous_well"])

    area_metrics = vc.evaluate_dispersion_prediction(initial_guess=np.asarray(dispersion_fit.result.x, dtype=float), held_out_area=np.asarray(held_out_no_treatment["area"], dtype=float), start_point=fold_start_point, params_yaml=fold_task["params_yaml"], regrowth_calibration=regrowth_fit)

    area_test_metrics = {key: value for key, value in area_metrics.items() if key.startswith("test_area_")}
    train_nutrient_metrics = {"train_calibration_nutrient_rmse": area_metrics.get("shared_nutrient_rmse", np.nan),
                              "train_calibration_nutrient_nrmse": area_metrics.get("shared_nutrient_nrmse", np.nan)}

    regrowth_test_metrics = vc.evaluate_regrowth_calibration_on_well(fold_task["held_out_continuous_well"], regrowth_fit, min_duration_frames=fold_task["min_duration_frames"], min_distance_um=fold_task["min_distance_um"])

    mutation_test_metrics = vc.evaluate_mutation_rate_prediction(mutation_rate=float(mutation_rate_fit.result.x[0]), held_out_clone_count=float(held_out_continuous["clone_count"]), params_yaml=fold_task["params_yaml"], prediction_replicas=fold_task["test_sim_replicas"], seed_offset=10000 * int(fold_task["fold_index"]), start_point_override=fold_start_point, simulation_param_overrides=fold_mutation_simulation_overrides)

    return {"fold_index": int(fold_task["fold_index"]),
            "held_out_no_treatment_label": fold_task["held_out_no_treatment_label"],
            "held_out_continuous_well": fold_task["held_out_continuous_well"],
            "train_no_treatment_count": len(train_no_treatment),
            "train_continuous_count": len(train_continuous),
            "regrowth_t0_h": float(regrowth_fit["t0_h"]),
            "regrowth_slope_um_per_h": float(regrowth_fit["slope_um_per_h"]),
            "fitted_diffusion_sensitive": float(dispersion_fit.result.x[0]),
            "fitted_uptake_rate": float(dispersion_fit.result.x[1]),
            "fitted_diffusion_nutrients": float(dispersion_fit.result.x[2]),
            "fitted_start_point": fold_start_point,
            "fitted_mutation_rate": float(mutation_rate_fit.result.x[0]),
            "regrowth_training_points": int(regrowth_fit["n_points"]),
            "dispersion_train_objective": float(dispersion_fit.result.fun),
            "dispersion_train_fit_converged": bool(dispersion_fit.result.success),
            "mutation_train_objective": float(mutation_rate_fit.result.fun),
            "mutation_train_fit_converged": bool(mutation_rate_fit.result.success),
            "test_clone_count": float(held_out_continuous["clone_count"]),
            **area_test_metrics,
            **train_nutrient_metrics,
            **regrowth_test_metrics,
            **mutation_test_metrics}


class EndToEndLoocvRunner:
    def __init__(self):
        self.config = self.load_config()
        self.params_yaml = vc.load_params_yaml('params.yaml')
        self.worker_count = self.config["num_workers"]

        self.dispersion_guess = np.array([self.params_yaml["diffusion_sensitive"], self.params_yaml["uptake_rate"], self.params_yaml["diffusion_nutrients"]], dtype=float)
        self.mutation_rate_guess = float(self.params_yaml["mutation_rate"])

        self.no_treatment_data = vc.load_no_treatment_area_dataset()
        self.continuous_therapy_data = vc.load_mutation_rate_dataset()
        self.continuous_therapy_wells = [str(entry["label"]) for entry in self.continuous_therapy_data]

    def load_config(self):
        config = vc.load_params_yaml(file_path="Validation/validation_params.yaml")
        config.setdefault("no_treatment_labels", None)
        config.setdefault("continuous_wells", None)
        config.setdefault("run_full_data_refit", False)
        config.setdefault("reference_parameter_overrides", {})
        config['output_dir'] = pl.Path(config['output_dir'])
        return config

    def build_reference_model_from_params(self):
        reference_model = {"diffusion_sensitive": float(self.params_yaml["diffusion_sensitive"]),
                           "uptake_rate": float(self.params_yaml["uptake_rate"]),
                           "diffusion_nutrients": float(self.params_yaml["diffusion_nutrients"]),
                           "start_point": int(self.params_yaml["start_point"]),
                           "mutation_rate": float(self.params_yaml["mutation_rate"])}

        regrowth_fit = vc.fit_regrowth_calibration_from_continuous_therapy(wells=self.continuous_therapy_wells, min_duration_frames=self.config["min_duration_frames"], min_distance_um=self.config["min_distance_um"])

        reference_model["regrowth_t0_h"] = float(regrowth_fit["t0_h"])
        reference_model["regrowth_slope_um_per_h"] = float(regrowth_fit["slope_um_per_h"])

        reference_model["reference_source"] = "params_yaml_reference"
        reference_model["fresh_refit_performed"] = False
        reference_model["regrowth_source"] = "full_continuous_therapy_calibration"
        return reference_model, {"regrowth_fit": regrowth_fit}

    def build_fold_tasks(self):
        fold_tasks = []
        fold_index = 0
        for no_treatment_entry in self.no_treatment_data:
            for continuous_entry in self.continuous_therapy_data:
                fold_index += 1
                fold_tasks.append({"fold_index": fold_index,
                                   "held_out_no_treatment_label": str(no_treatment_entry["label"]),
                                   "held_out_continuous_well": str(continuous_entry["label"]),
                                   "no_treatment_data": self.no_treatment_data,
                                   "continuous_therapy_data": self.continuous_therapy_data,
                                   "params_yaml": self.params_yaml,
                                   "dispersion_guess": self.dispersion_guess,
                                   "mutation_rate_guess": self.mutation_rate_guess,
                                   "maxiter_dispersion": self.config["maxiter_dispersion"],
                                   "maxiter_mutation_rate": self.config["maxiter_mutation_rate"],
                                   "fit_sim_replicas": self.config["fit_sim_replicas"],
                                   "test_sim_replicas": self.config["test_sim_replicas"],
                                   "min_duration_frames": self.config["min_duration_frames"],
                                   "min_distance_um": self.config["min_distance_um"]})
        return fold_tasks

    def run_folds(self, fold_tasks):
        total_folds = len(fold_tasks)
        fold_rows = []

        if self.worker_count == 1:
            for fold_task in fold_tasks:
                self.logger.info("Fold %d/%d holding out no-treatment=%s and continuous-therapy=%s",
                                 fold_task["fold_index"], total_folds,
                                 fold_task["held_out_no_treatment_label"],
                                 fold_task["held_out_continuous_well"])
                fold_rows.append(run_end_to_end_fold(fold_task))
        else:
            self.logger.info("Submitting %d fold task(s) to the process pool. The first completed fold may take hours because each fold reruns the full hierarchy.", total_folds)
            with cf.ProcessPoolExecutor(max_workers=self.worker_count) as executor:
                future_map = {executor.submit(run_end_to_end_fold, fold_task): fold_task for fold_task in fold_tasks}
                self.logger.info("All fold tasks submitted; waiting for completed folds.")
                for future in cf.as_completed(future_map):
                    fold_task = future_map[future]
                    try:
                        row = future.result()
                    except Exception:
                        self.logger.exception("Fold %d/%d crashed while holding out no-treatment=%s and continuous-therapy=%s",
                                              fold_task["fold_index"], total_folds,
                                              fold_task["held_out_no_treatment_label"],
                                              fold_task["held_out_continuous_well"])
                        raise
                    self.logger.info("Completed fold %d/%d holding out no-treatment=%s and continuous-therapy=%s",
                                     fold_task["fold_index"], total_folds,
                                     fold_task["held_out_no_treatment_label"],
                                     fold_task["held_out_continuous_well"])
                    fold_rows.append(row)

        fold_rows.sort(key=lambda row: row["fold_index"])
        return pd.DataFrame(fold_rows)

    def make_parameter_summary(self, fold_df, reference_model):
        parameter_summary = vc.build_parameter_summary(fold_df=fold_df, parameter_columns=get_parameter_columns(), 
                                                       full_fit_values={"regrowth_t0_h": float(reference_model["regrowth_t0_h"]),
                                                                        "regrowth_slope_um_per_h": float(reference_model["regrowth_slope_um_per_h"]),
                                                                        "fitted_diffusion_sensitive": float(reference_model["diffusion_sensitive"]),
                                                                        "fitted_uptake_rate": float(reference_model["uptake_rate"]),
                                                                        "fitted_diffusion_nutrients": float(reference_model["diffusion_nutrients"]),
                                                                        "fitted_start_point": float(reference_model["start_point"]),
                                                                        "fitted_mutation_rate": float(reference_model["mutation_rate"])})
        
        parameter_summary["reference_source"] = str(reference_model["reference_source"])
        parameter_summary["fresh_refit_performed"] = bool(reference_model["fresh_refit_performed"])
        return parameter_summary

    def save_outputs(self, run_dir, fold_df, parameter_summary, headline_summary, metric_summary, reference_model, reference_artifacts):
        fold_df.to_csv(run_dir / "fold_results.csv", index=False)
        parameter_summary.to_csv(run_dir / "parameter_summary.csv", index=False)
        headline_summary.to_csv(run_dir / "headline_summary.csv", index=False)
        with (run_dir / "headline_summary.txt").open("w", encoding="utf-8") as handle:
            handle.write(build_headline_summary_text(headline_summary))

        vc.save_json(run_dir / "summary.json",
                        {"validation_strategy": "hierarchical paired leave-one-out cross-validation across both data sources",
                        "hierarchy": ["1. Fit regrowth calibration on continuous-therapy training wells.",
                                    "2. Fit diffusion, nutrient parameters, and start_point on no-treatment training trajectories using that regrowth calibration.",
                                    "3. Fit mutation_rate on continuous-therapy training wells using the fold-specific diffusion, uptake, nutrient diffusion, and start_point from step 2.",
                                    "4. Evaluate each fold on the held-out no-treatment trajectory and the held-out continuous-therapy well."],
                        "n_no_treatment_trajectories": len(self.no_treatment_data),
                        "n_continuous_wells": len(self.continuous_therapy_data),
                        "n_folds": len(fold_df),
                        "full_data_fit": {"regrowth_t0_h": float(reference_model["regrowth_t0_h"]),
                                        "regrowth_slope_um_per_h": float(reference_model["regrowth_slope_um_per_h"]),
                                        "diffusion_sensitive": float(reference_model["diffusion_sensitive"]),
                                        "uptake_rate": float(reference_model["uptake_rate"]),
                                        "diffusion_nutrients": float(reference_model["diffusion_nutrients"]),
                                        "start_point": int(reference_model["start_point"]),
                                        "mutation_rate": float(reference_model["mutation_rate"]),
                                        "reference_source": str(reference_model["reference_source"]),
                                        "fresh_refit_performed": bool(reference_model["fresh_refit_performed"]),
                                        "regrowth_source": str(reference_model.get("regrowth_source", "fresh_full_data_refit")),
                                        "dispersion_objective": float(reference_artifacts["dispersion_fit"].result.fun) if "dispersion_fit" in reference_artifacts else None,
                                        "mutation_objective": float(reference_artifacts["mutation_fit"].result.fun) if "mutation_fit" in reference_artifacts else None},
                        "metric_summary": metric_summary,
                        "notes": ["This workflow keeps the fold-specific regrowth, diffusion, uptake, nutrient diffusion, and start_point values in memory; it never rewrites params.yaml.",
                                "Each fold holds out one no-treatment trajectory and one continuous-therapy well simultaneously.",
                                "Regrowth calibration, dispersion fitting, and mutation-rate fitting all use only the training data for that fold and pass upstream fitted values downstream in memory.",
                                "train_calibration_nutrient_* columns are reported only as training-side consistency checks against the fold-specific regrowth calibration; the held-out test metrics are test_area_*, test_regrowth_*, and mutation-rate test errors.",
                                "headline_summary.csv and headline_summary.txt provide a compact report-style view of the main held-out metrics and parameter-stability summaries.",
                                "Parameter stability across the full hierarchical pipeline is summarized in parameter_summary.csv.",
                                "When run_full_data_refit is false, the reference values in parameter_summary.csv come from params.yaml plus any config overrides, with regrowth taken from either the override or a cheap full continuous-therapy calibration."]})
        vc.save_json(run_dir / "run_config.json",
                        {"dispersion_initial": self.dispersion_guess,
                        "mutation_rate_initial": self.mutation_rate_guess,
                        "maxiter_dispersion": self.config["maxiter_dispersion"],
                        "maxiter_mutation_rate": self.config["maxiter_mutation_rate"],
                        "fit_sim_replicas": self.config["fit_sim_replicas"],
                        "test_sim_replicas": self.config["test_sim_replicas"],
                        "min_duration_frames": self.config["min_duration_frames"],
                        "min_distance_um": self.config["min_distance_um"],
                        "no_treatment_labels": self.config["no_treatment_labels"],
                        "continuous_wells": self.config["continuous_wells"],
                        "num_workers": self.worker_count,
                        "run_full_data_refit": bool(self.config["run_full_data_refit"]),
                        "reference_parameter_overrides": self.config["reference_parameter_overrides"],
                        "output_dir": run_dir})

    def run(self):
        run_dir = vc.create_run_directory(self.config["output_dir"], "end_to_end_loocv")
        self.logger = vc.setup_logger(run_dir / "validation.log")

        self.logger.info("Running end-to-end LOOCV with %d no-treatment trajectories and %d continuous-therapy wells.", len(self.no_treatment_data), len(self.continuous_therapy_data))
        self.logger.info("Using %d worker process(es) for fold evaluation.", self.worker_count)

        reference_model, reference_artifacts = self.build_reference_model_from_params()
        self.logger.info("Reference model ready: start_point=%s diffusion=%s uptake=%s nutrient_diffusion=%s mutation_rate=%s",
                         reference_model["start_point"], reference_model["diffusion_sensitive"],
                         reference_model["uptake_rate"], reference_model["diffusion_nutrients"],
                         reference_model["mutation_rate"])

        fold_tasks = self.build_fold_tasks()
        self.logger.info("Prepared %d end-to-end fold task(s).", len(fold_tasks))
        fold_df = self.run_folds(fold_tasks)

        parameter_summary = self.make_parameter_summary(fold_df, reference_model)
        metric_summary = vc.summarize_fold_metrics(fold_df, get_metric_columns())
        headline_summary = build_headline_summary(metric_summary, parameter_summary)
        self.save_outputs(run_dir, fold_df, parameter_summary, headline_summary, metric_summary, reference_model, reference_artifacts)
        self.logger.info("Saved end-to-end LOOCV results to %s", run_dir)
        return run_dir


def main():
    EndToEndLoocvRunner().run()


if __name__ == "__main__":
    main()
