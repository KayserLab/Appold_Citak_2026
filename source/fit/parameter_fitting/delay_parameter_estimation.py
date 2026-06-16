from pathlib import Path
import numpy as np
import pandas as pd


class DelayParameterEstimation:
    def __init__(self, no_treatment_dir=None, pulse_dir=None, met_7_dir=None, pulse_length=259, met_7_length=302, treatment_times=(36, 64), output_scale=10.0, time_step_h=0.5, no_treatment_pixel_size_um=8.648):
        self.project_root = Path(__file__).resolve().parents[3]
        self.paths = {"no_treatment_dir": (f'{self.project_root}/data/exp_data/No_treatment_control' if no_treatment_dir is None else Path(no_treatment_dir)),
                      "pulse_dir": (f'{self.project_root}/data/exp_data/14h_Pulse_no_clone_radius' if pulse_dir is None else Path(pulse_dir)),
                      "met_7_dir": (f'{self.project_root}/data/exp_data/7h_18h_no_clone_radius' if met_7_dir is None else Path(met_7_dir))}
        self.params = {"pulse_length": None if pulse_length is None else int(pulse_length),
                       "met_7_length": None if met_7_length is None else int(met_7_length),
                       "treatment_times": tuple(treatment_times),
                       "output_scale": float(output_scale),
                       "time_step_h": float(time_step_h),
                       "no_treatment_pixel_size_um": float(no_treatment_pixel_size_um)}

    def rolling_median(self, data, window_size):
        half_window = int(window_size) // 2
        padded_data = np.pad(
            np.asarray(data, dtype=float),
            (half_window, half_window),
            mode="edge",
        )
        return np.array(
            [
                np.median(padded_data[index : index + int(window_size)])
                for index in range(len(data))
            ],
            dtype=float,
        )

    def fit_constant_ls(self, y):
        y = np.asarray(y, dtype=float)
        level = float(np.mean(y))
        residuals = y - level
        dof = y.size - 1
        sse = float(np.sum(residuals**2))
        sigma2 = sse / dof if dof > 0 else np.nan
        cov = np.array([[sigma2 / y.size]]) if np.isfinite(sigma2) else np.array([[np.nan]])
        return {"coef": np.array([level], dtype=float),
                "cov": cov,
                "stderr": np.array([np.sqrt(cov[0, 0])], dtype=float),
                "sse": sse,
                "dof": dof}

    def fit_linear_ls(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.vstack([x, np.ones_like(x)]).T
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ beta
        dof = y.size - 2
        sse = float(np.sum(residuals**2))

        if dof > 0:
            sigma2 = sse / dof
            cov = sigma2 * np.linalg.inv(design.T @ design)
            stderr = np.sqrt(np.diag(cov))
        else:
            cov = np.full((2, 2), np.nan)
            stderr = np.full(2, np.nan)

        return {"coef": beta.astype(float),
                "cov": cov,
                "stderr": stderr.astype(float),
                "sse": sse,
                "dof": dof}

    def safe_divide(self, numerator, denominator, context):
        if np.isclose(denominator, 0.0):
            raise ValueError(f"{context} is undefined because the denominator is zero.")
        return numerator / denominator

    def intersect_constant_with_line(self, constant_level, line_fit):
        slope, intercept = np.asarray(line_fit["coef"], dtype=float)
        return self.safe_divide(float(constant_level) - intercept, slope, "Constant/line intersection")

    def intersect_two_lines(self, line_fit_1, line_fit_2):
        slope_1, intercept_1 = np.asarray(line_fit_1["coef"], dtype=float)
        slope_2, intercept_2 = np.asarray(line_fit_2["coef"], dtype=float)
        return self.safe_divide(intercept_2 - intercept_1, slope_1 - slope_2, "Line/line intersection")

    def combine_independent_variances(self, *gradient_covariance_pairs):
        variance = 0.0
        for gradient, covariance in gradient_covariance_pairs:
            gradient = np.asarray(gradient, dtype=float)
            covariance = np.asarray(covariance, dtype=float)
            variance += float(gradient @ covariance @ gradient)
        return variance

    def predict_4seg(self, x, a, b, c, y0, m1, q1, m3):
        x = np.asarray(x, dtype=float)
        yhat = np.empty_like(x, dtype=float)
        y2 = m1 * b + q1

        before_mask = x < a
        rising_mask = (x >= a) & (x < b)
        plateau_mask = (x >= b) & (x < c)
        recovery_mask = x >= c

        yhat[before_mask] = y0
        yhat[rising_mask] = m1 * x[rising_mask] + q1
        yhat[plateau_mask] = y2
        yhat[recovery_mask] = y2 + m3 * (x[recovery_mask] - c)
        return yhat

    def grid_fit_4seg_once(self, x, y, m3, linear_fit_end=None, min_len=(2, 3, 3, 2), collect_candidates=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        unique_x = np.unique(x)
        min0, min1, min2, min3len = min_len
        best = {"sse": np.inf}
        candidates = [] if collect_candidates else None

        for ia in range(min0, len(unique_x) - (min1 + min2 + min3len)):
            a_value = unique_x[ia]
            y0 = float(np.mean(y[x < a_value]))

            for ib in range(ia + min1, len(unique_x) - (min2 + min3len)):
                b_value = unique_x[ib]

                for ic in range(ib + min2, len(unique_x) - min3len + 1):
                    c_value = unique_x[ic]
                    linear_mask = (x >= a_value) & (x < b_value)
                    if linear_fit_end is not None:
                        linear_mask = (x >= a_value) & (x < min(b_value, linear_fit_end))

                    if np.sum(linear_mask) < 2:
                        continue

                    line_fit = self.fit_linear_ls(x[linear_mask], y[linear_mask])
                    m1, q1 = line_fit["coef"]
                    yhat = self.predict_4seg(x, a_value, b_value, c_value, y0, m1, q1, m3)
                    sse = float(np.sum((y - yhat) ** 2))
                    candidate = {"a": float(a_value),
                                 "b": float(b_value),
                                 "c": float(c_value),
                                 "y0": y0,
                                 "m1": float(m1),
                                 "q1": float(q1),
                                 "sse": sse}
                    if collect_candidates:
                        candidates.append(candidate)
                    if sse < best["sse"]:
                        best = candidate

        if collect_candidates:
            return best, candidates
        return best

    def summarize_grid_fit_uncertainty(self, candidates, n_points, n_parameters=6):
        if len(candidates) == 0:
            return {"stderr": {},
                    "weighted_mean": {},
                    "sigma2": np.nan,
                    "dof": 0,
                    "n_candidates": 0,
                    "weights": np.asarray([], dtype=float)}

        sse = np.array([candidate["sse"] for candidate in candidates], dtype=float)
        sse_min = float(np.min(sse))
        dof = max(int(n_points - n_parameters), 1)
        sigma2 = sse_min / dof if sse_min > 0 else np.finfo(float).eps

        log_weights = -(sse - sse_min) / (2 * sigma2)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        param_names = ("a", "b", "c", "y0", "m1", "q1")
        weighted_mean = {}
        stderr = {}
        for name in param_names:
            values = np.array([candidate[name] for candidate in candidates], dtype=float)
            mean = float(np.sum(weights * values))
            variance = float(np.sum(weights * (values - mean) ** 2))
            weighted_mean[name] = mean
            stderr[name] = np.sqrt(max(variance, 0.0))

        return {"stderr": stderr,
                "weighted_mean": weighted_mean,
                "sigma2": float(sigma2),
                "dof": dof,
                "n_candidates": len(candidates),
                "weights": weights.astype(float)}

    def grid_fit_4seg(self, x, y, m3, linear_fit_end=None, min_len=(2, 3, 3, 2)):
        best, candidates = self.grid_fit_4seg_once(x, y, m3=m3, linear_fit_end=linear_fit_end, min_len=min_len, collect_candidates=True)
        uncertainty = self.summarize_grid_fit_uncertainty(candidates, n_points=len(np.asarray(y)))
        best["stderr"] = uncertainty["stderr"]
        best["weighted_mean"] = uncertainty["weighted_mean"]
        best["uncertainty_method"] = "profile_likelihood_from_grid_sse"
        best["grid_sigma2"] = uncertainty["sigma2"]
        best["grid_dof"] = uncertainty["dof"]
        best["n_candidates"] = uncertainty["n_candidates"]
        best["weights"] = uncertainty["weights"]
        best["candidates"] = candidates
        return best

    def load_pulse_data(self, folder_path, length=None):
        sus_dir = Path(folder_path) / "Sus_Kymos"
        csv_paths = sorted(sus_dir.glob("*max_distance.csv"))
        radial_traces = []
        for csv_path in csv_paths:
            data = pd.read_csv(csv_path)
            radial_traces.append(self.rolling_median(data["max_distance_mm"].to_numpy(dtype=float)[:length], window_size=9))

        radial_mean = np.mean(radial_traces, axis=0)
        radial_derivative = np.gradient(radial_mean, self.params["time_step_h"])
        return radial_mean, radial_derivative

    def load_no_treatment_data(self, folder_path):
        csv_paths = sorted(Path(folder_path).glob("colony*clonearea.csv"))
        return [pd.read_csv(csv_path) for csv_path in csv_paths]

    def median_over_radius_derivative_no_treatment(self, exp_data):
        radius_derivatives = []
        radii = []
        for exp in exp_data:
            colony_radius = (
                exp["colony_radius"].to_numpy(dtype=float)
                * self.params["no_treatment_pixel_size_um"]
                / 1e3
            )
            radius_derivatives.append(np.gradient(colony_radius, self.params["time_step_h"])[:295])
            radii.append(colony_radius[:295])
        return (
            self.rolling_median(np.median(np.array(radius_derivatives), axis=0), window_size=9),
            self.rolling_median(np.median(np.array(radii), axis=0), window_size=9),
        )

    def load_delay_analysis_inputs(self):
        no_treatment_dataset = self.load_no_treatment_data(self.paths["no_treatment_dir"])
        nt_derivative, nt_radius = self.median_over_radius_derivative_no_treatment(no_treatment_dataset)
        pulse_mean, pulse_derivative = self.load_pulse_data(self.paths["pulse_dir"], length=self.params["pulse_length"])
        met_7_mean, met_7_derivative = self.load_pulse_data(self.paths["met_7_dir"], length=self.params["met_7_length"])
        return {"nt_derivative": nt_derivative,
                "nt_radius": nt_radius,
                "pulse_mean": pulse_mean,
                "pulse_derivative": pulse_derivative,
                "met_7_mean": met_7_mean,
                "met_7_derivative": met_7_derivative}

    def analyze_delay_parameters(self):
        data = self.load_delay_analysis_inputs()
        pulse_derivative = data["pulse_derivative"]
        met_7_derivative = data["met_7_derivative"]
        nt_derivative = data["nt_derivative"]

        pulse_constant_fit = self.fit_constant_ls(pulse_derivative[70:85])
        pulse_line_fit = self.fit_linear_ls(np.arange(len(pulse_derivative[36:65])) + 36, pulse_derivative[36:65])
        met_7_constant_fit = self.fit_constant_ls(met_7_derivative[60:75])
        release_line_fit = self.fit_linear_ls(np.arange(len(pulse_derivative[103:115])) + 103, pulse_derivative[103:115])
        control_line_fit = self.fit_linear_ls(np.arange(len(nt_derivative[90:125])) + 90, nt_derivative[90:125])

        treatment_start = float(self.params["treatment_times"][0])
        output_scale = self.params["output_scale"]

        treat_on_cross = self.intersect_constant_with_line(pulse_constant_fit["coef"][0], pulse_line_fit)
        treatment_delay = (treat_on_cross - treatment_start) * output_scale
        treatment_delay_legacy = np.round(treat_on_cross - treatment_start) * output_scale

        slope_on, intercept_on = pulse_line_fit["coef"]
        constant_on = pulse_constant_fit["coef"][0]
        treat_on_gradient_line = np.array([-(constant_on - intercept_on) / (slope_on**2), -1.0 / slope_on], dtype=float)
        treat_on_gradient_constant = np.array([1.0 / slope_on], dtype=float)
        treat_on_variance = self.combine_independent_variances((treat_on_gradient_line, pulse_line_fit["cov"]), (treat_on_gradient_constant, pulse_constant_fit["cov"]))
        treatment_delay_stderr = output_scale * np.sqrt(max(treat_on_variance, 0.0))

        treat_off_release_cross = self.intersect_constant_with_line(pulse_constant_fit["coef"][0], release_line_fit)
        treat_off_recovery_cross = self.intersect_two_lines(release_line_fit, control_line_fit)
        release_delay = (treat_off_recovery_cross - treat_off_release_cross) * output_scale
        release_delay_legacy = (np.round(treat_off_recovery_cross - treat_off_release_cross) * output_scale)

        slope_release, intercept_release = release_line_fit["coef"]
        slope_control, intercept_control = control_line_fit["coef"]
        denominator = slope_release - slope_control
        numerator = intercept_control - intercept_release
        release_constant_gap = pulse_constant_fit["coef"][0] - intercept_release
        treat_off_gradient_release = np.array([-(numerator / (denominator**2)) + release_constant_gap / (slope_release**2), -1.0 / denominator + 1.0 / slope_release], dtype=float)
        treat_off_gradient_control = np.array([numerator / (denominator**2), 1.0 / denominator], dtype=float)
        treat_off_gradient_constant = np.array([-1.0 / slope_release], dtype=float)
        treat_off_variance = self.combine_independent_variances((treat_off_gradient_release, release_line_fit["cov"]), (treat_off_gradient_control, control_line_fit["cov"]), (treat_off_gradient_constant, pulse_constant_fit["cov"]))
        release_delay_stderr = output_scale * np.sqrt(max(treat_off_variance, 0.0))

        xdata = np.arange(36, np.argmax(met_7_derivative[60:110]) + 60, dtype=float)
        ydata = met_7_derivative[36 : np.argmax(met_7_derivative[60:110]) + 60].astype(float)
        recovery_slope = float(release_line_fit["coef"][0])
        four_segment_fit = self.grid_fit_4seg(xdata, ydata, m3=recovery_slope, linear_fit_end=55)

        overshoot_steps = (four_segment_fit["b"] - treatment_start - 14.0) * output_scale  # 7h pulse duration -> 14 steps at 0.5h per step
        overshoot_steps_stderr = four_segment_fit["stderr"]["b"] * output_scale
        lag_steps = (four_segment_fit["c"] - treatment_start - 14.0) * output_scale  # 7h pulse duration -> 14 steps at 0.5h per step
        lag_steps_stderr = four_segment_fit["stderr"]["c"] * output_scale

        xfit = np.linspace(xdata.min(), xdata.max(), 500)
        yfit = self.predict_4seg(xfit, four_segment_fit["a"], four_segment_fit["b"], four_segment_fit["c"], four_segment_fit["y0"], four_segment_fit["m1"], four_segment_fit["q1"], m3=recovery_slope)

        return {"treatment_times": tuple(int(value) for value in self.params["treatment_times"]),
                "output_scale": float(output_scale),
                "data": data,
                "fits": {"pulse_constant": pulse_constant_fit,
                         "pulse_line": pulse_line_fit,
                         "met_7_constant": met_7_constant_fit,
                         "release_line": release_line_fit,
                         "control_line": control_line_fit,
                         "four_segment": four_segment_fit},
                "four_segment_xdata": xdata,
                "four_segment_ydata": ydata,
                "four_segment_xfit": xfit,
                "four_segment_yfit": yfit,
                "parameter_estimates": {"treatment_delay": {"estimate": float(treatment_delay),
                                                            "stderr": float(treatment_delay_stderr),
                                                            "legacy_rounded": float(treatment_delay_legacy)},
                                        "release_delay": {"estimate": float(release_delay),
                                                          "stderr": float(release_delay_stderr),
                                                          "legacy_rounded": float(release_delay_legacy)},
                                        "overshoot_steps": {"estimate": float(overshoot_steps),
                                                            "stderr": float(overshoot_steps_stderr),
                                                            "legacy_rounded": float(np.round(overshoot_steps / output_scale) * output_scale)},
                                        "lag_steps": {"estimate": float(lag_steps),
                                                      "stderr": float(lag_steps_stderr),
                                                      "legacy_rounded": float(np.round(lag_steps / output_scale) * output_scale)}}}
    

    def sample_fit_coefficients(self, fit_result, rng):
        mean = np.asarray(fit_result["coef"], dtype=float)
        cov = np.asarray(fit_result["cov"], dtype=float)
        if mean.size == 1:
            variance = float(cov[0, 0]) if cov.shape == (1, 1) and np.isfinite(cov[0, 0]) else np.nan
            if not np.isfinite(variance) or variance <= 0:
                return mean.copy()
            return np.array([rng.normal(float(mean[0]), np.sqrt(variance))], dtype=float)

        if cov.shape != (mean.size, mean.size) or not np.all(np.isfinite(cov)):
            return mean.copy()

        try:
            return np.asarray(rng.multivariate_normal(mean, cov), dtype=float)
        except Exception:
            return mean.copy()

    def sample_delay_parameter_sets(self, analysis, n_samples, rng, center_values=None):
        n_samples = int(n_samples)
        parameter_estimates = analysis["parameter_estimates"]
        if center_values is None:
            center_values = {key: float(spec["estimate"]) for key, spec in parameter_estimates.items()}
        else:
            center_values = {key: float(value) for key, value in center_values.items()}

        shift_map = {key: float(center_values[key]) - float(parameter_estimates[key]["estimate"]) for key in parameter_estimates}
        treatment_start = float(analysis["treatment_times"][0])
        output_scale = float(analysis["output_scale"])
        pulse_constant_fit = analysis["fits"]["pulse_constant"]
        pulse_line_fit = analysis["fits"]["pulse_line"]
        release_line_fit = analysis["fits"]["release_line"]
        control_line_fit = analysis["fits"]["control_line"]
        four_segment_fit = analysis["fits"]["four_segment"]
        candidates = four_segment_fit.get("candidates", [])
        weights = np.asarray(four_segment_fit.get("weights", []), dtype=float)

        treatment_delay_samples = np.full(n_samples, np.nan, dtype=float)
        release_delay_samples = np.full(n_samples, np.nan, dtype=float)
        for sample_idx in range(n_samples):
            sampled_treatment_delay = np.nan
            sampled_release_delay = np.nan
            for _ in range(64):
                constant_draw = float(self.sample_fit_coefficients(pulse_constant_fit, rng)[0])
                pulse_line_draw = self.sample_fit_coefficients(pulse_line_fit, rng)
                release_line_draw = self.sample_fit_coefficients(release_line_fit, rng)
                control_line_draw = self.sample_fit_coefficients(control_line_fit, rng)
                try:
                    treatment_cross = self.intersect_constant_with_line(constant_draw, {"coef": pulse_line_draw})
                    release_cross = self.intersect_constant_with_line(constant_draw, {"coef": release_line_draw})
                    recovery_cross = self.intersect_two_lines({"coef": release_line_draw}, {"coef": control_line_draw})
                except ValueError:
                    continue

                sampled_treatment_delay = (treatment_cross - treatment_start) * output_scale
                sampled_release_delay = (recovery_cross - release_cross) * output_scale
                if (np.isfinite(sampled_treatment_delay) and np.isfinite(sampled_release_delay) and sampled_treatment_delay > 0 and sampled_release_delay > 0):
                    break

            if not np.isfinite(sampled_treatment_delay) or sampled_treatment_delay <= 0:
                sampled_treatment_delay = float(center_values["treatment_delay"])
            else:
                sampled_treatment_delay += shift_map["treatment_delay"]

            if not np.isfinite(sampled_release_delay) or sampled_release_delay <= 0:
                sampled_release_delay = float(center_values["release_delay"])
            else:
                sampled_release_delay += shift_map["release_delay"]

            treatment_delay_samples[sample_idx] = sampled_treatment_delay
            release_delay_samples[sample_idx] = sampled_release_delay

        overshoot_samples = np.full(n_samples, float(center_values["overshoot_steps"]), dtype=float)
        lag_samples = np.full(n_samples, float(center_values["lag_steps"]), dtype=float)
        if (candidates and weights.size == len(candidates) and np.all(np.isfinite(weights)) and np.sum(weights) > 0):
            candidate_indices = rng.choice(len(candidates), size=n_samples, replace=True, p=weights / np.sum(weights))
            selected_candidates = [candidates[int(index)] for index in candidate_indices]
            overshoot_samples = np.array([(float(candidate["b"]) - treatment_start - 14.0) * output_scale + shift_map["overshoot_steps"] for candidate in selected_candidates], dtype=float)
            lag_samples = np.array([(float(candidate["c"]) - treatment_start - 14.0) * output_scale + shift_map["lag_steps"] for candidate in selected_candidates], dtype=float)

        return {"treatment_delay": treatment_delay_samples,
                "release_delay": release_delay_samples,
                "overshoot_steps": overshoot_samples,
                "lag_steps": lag_samples}


def predict_4seg(x, a, b, c, y0, m1, q1, m3):
    delay_estimation = DelayParameterEstimation()
    return delay_estimation.predict_4seg(x, a, b, c, y0, m1, q1, m3)


def grid_fit_4seg(x, y, m3, linear_fit_end=None, min_len=(2, 3, 3, 2)):
    delay_estimation = DelayParameterEstimation()
    return delay_estimation.grid_fit_4seg(
        x,
        y,
        m3,
        linear_fit_end=linear_fit_end,
        min_len=min_len,
    )


def load_delay_analysis_inputs(
    *,
    no_treatment_dir=None,
    pulse_dir=None,
    met_7_dir=None,
    pulse_length=259,
    met_7_length=302,
):
    delay_estimation = DelayParameterEstimation(
        no_treatment_dir=no_treatment_dir,
        pulse_dir=pulse_dir,
        met_7_dir=met_7_dir,
        pulse_length=pulse_length,
        met_7_length=met_7_length,
    )
    return delay_estimation.load_delay_analysis_inputs()


def analyze_delay_parameters(no_treatment_dir=None, pulse_dir=None, met_7_dir=None, pulse_length=259, met_7_length=302, treatment_times=(36, 64), output_scale=10.0):
    delay_estimation = DelayParameterEstimation(no_treatment_dir=no_treatment_dir, pulse_dir=pulse_dir, met_7_dir=met_7_dir, pulse_length=pulse_length, met_7_length=met_7_length, treatment_times=treatment_times, output_scale=output_scale)
    return delay_estimation.analyze_delay_parameters()


def sample_delay_parameter_sets(analysis, n_samples, rng, *, center_values=None):
    delay_estimation = DelayParameterEstimation()
    return delay_estimation.sample_delay_parameter_sets(analysis, n_samples, rng, center_values=center_values)
