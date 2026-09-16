import sys, json, glob, re, os
import logging
import traceback

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

from datetime import datetime, timezone

from config import PATH

# Configure logging
logger = logging.getLogger(__name__)

# Cached training traces for anomaly detection / deadline thresholds
_training_traces_df = None
iso_forest = None
iso_forest_features = None

# read stat history
def read_history(label, base_label):
    history_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_stats_history.csv')
    history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')
    
    # Remove the "Name" column which contains "Aggregated" string values
    if "Name" in history_df.columns:
        history_df = history_df.drop(columns=["Name"])

    return history_df

# read response time data
def read_response(label, base_label):
    resp_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_responce_log.csv')
    
    return resp_df

# read Jaeger trace data
def read_traces(path):
    data = {}
    with open(path, 'r') as file:
        data = json.load(file)

    rows = []
    for trace in data["data"]:
        row = {"id": trace['traceID']}
        
        total = 0
        st = sys.maxsize
        for s in trace["spans"]:
            st = min(st, s["startTime"])
            total += s["duration"]
            row[s["operationName"]] =  s["duration"]
        
        row["startTime"] = st
        row["total"] = total
        rows.append(row)

    # fill NaN
    trace_df = pd.DataFrame(rows).fillna(0)

    # convert times
    trace_df["startTime"] = pd.to_datetime(trace_df['startTime'], unit='us')

    # compute binary pattern in a shared, consistent way
    trace_df = compute_trace_pattern(trace_df)

    return trace_df


def compute_trace_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the binary pattern column from span-duration columns in a
    consistent way for both training traces and live traces.

    We treat span columns as all columns except id/startTime/total/pattern.
    """
    if df is None or df.empty:
        return df

    span_cols = df.columns.difference(["id", "startTime", "total", "pattern"])
    if not len(span_cols):
        logger.debug("compute_trace_pattern: no span columns found, returning original dataframe")
        return df

    df = df.copy()
    df["pattern"] = df[span_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)
    return df


def align_traces_to_training_schema(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align span feature columns in a live trace dataframe to match the
    training traces schema, so that patterns are comparable.
    - Uses the same span columns as the cached training traces.
    - Adds missing span columns as 0.
    - Drops extra span columns that don't exist in training.
    - Recomputes the binary pattern string over the aligned span columns.
    """
    training_df = _load_training_traces()
    # In training traces, span columns are everything except id/startTime/total.
    train_span_cols = training_df.columns.difference(["id", "startTime", "total", "pattern"])
    logger.debug(
        "align_traces_to_training_schema: training span columns (%d): %s",
        len(train_span_cols),
        list(train_span_cols),
    )

    # Ensure all training span columns exist in the live dataframe.
    for col in train_span_cols:
        if col not in trace_df.columns:
            trace_df[col] = 0

    # Drop extra span columns that are not part of the training schema.
    live_span_cols = trace_df.columns.difference(["id", "startTime", "total", "pattern"])
    extra_cols = [c for c in live_span_cols if c not in list(train_span_cols)]
    if extra_cols:
        logger.debug(
            "align_traces_to_training_schema: dropping extra live span columns: %s",
            extra_cols,
        )
        trace_df = trace_df.drop(columns=extra_cols)

    # Reorder columns so that span columns follow the training order.
    span_cols = list(train_span_cols)
    front_cols = ["id", "startTime", "total"]
    ordered_cols = [c for c in front_cols if c in trace_df.columns] + span_cols
    remaining = [c for c in trace_df.columns if c not in ordered_cols]
    trace_df = trace_df[ordered_cols + remaining]

    # Recompute pattern using the shared helper so semantics
    # stay identical to training traces.
    trace_df = compute_trace_pattern(trace_df)
    if "pattern" in trace_df.columns:
        logger.debug(
            "align_traces_to_training_schema: unique live patterns after alignment: %s",
            trace_df["pattern"].unique(),
        )

    return trace_df


def _load_training_traces():
    """
    Load and cache training traces from CSV or JSON for use by
    anomaly detection and deadline calculations.
    """
    global _training_traces_df
    if _training_traces_df is not None:
        return _training_traces_df

    json_path = os.path.join(PATH, "anomaly_detection", "traces-1000u-refined.json")

    logger.info(f"Loading training traces found: {json_path}")
    df = read_traces(json_path)

    # Ensure patterns exist and are computed with the same semantics
    # as for live traces.
    df = compute_trace_pattern(df)

    _training_traces_df = df
    return _training_traces_df


def read_metrics(label, base_label):
    metrics = {}
    for n in glob.glob(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_*_metrics.csv'):
        match = re.search(r"[^_/]+-[^_]+(?=_metrics\.csv)", n)
        name = "unknown"
        if match:
            name=match.group()
        else:
            # logger.warning(f"unrecognized file: {n}, skipping")
            continue

        metric_df = pd.read_csv(n, index_col=False).drop('Unnamed: 0', axis=1)
        metric_columns = metric_df.columns.drop('index')
        metric_df["index"] = pd.to_datetime(metric_df['index'], unit='s')

        metrics[name] = metric_df
    
    return metrics


def shap_decisions(iso_forest, features):
    # try to explain a specific data points. as a short cut for RCA.
    shap_values = shap.TreeExplainer(iso_forest).shap_values(features)

    return shap_values, features.columns.tolist()


def label_anomalies_with_isolation_forest(trace_df, pretrained_features=None):
    """
    Train an IsolationForest on numeric trace features, compute anomaly labels,
    and return a boolean mask for anomalies.

    This is a port of the helper from `analysis/10-training-trace/trace-analysis.py`
    so that anomaly detection semantics stay consistent across scripts.
    """
    if trace_df is None or trace_df.empty:
        return np.array([], dtype=bool), None, []

    feature_columns = trace_df.select_dtypes(include=["number"]).columns.tolist()

    if not feature_columns:
        return np.zeros(len(trace_df), dtype=bool), None, []

    if "pattern" not in trace_df.columns:
        patterns = [None]
    else:
        patterns = list(trace_df["pattern"].unique())

    anomaly_mask = np.zeros(len(trace_df), dtype=bool)

    for pattern in patterns:
        if pattern is None:
            df_pat = trace_df
        else:
            df_pat = trace_df[trace_df["pattern"] == pattern]

        if df_pat.empty:
            continue

        feat_pat = df_pat[feature_columns].select_dtypes(include=["number"]).copy()

        if pretrained_features is not None and not pretrained_features.empty:
            train_df = pretrained_features.select_dtypes(include=["number"]).copy()
            if "pattern" in pretrained_features.columns and pattern is not None:
                train_df = pretrained_features[pretrained_features["pattern"] == pattern]
                train_df = train_df.select_dtypes(include=["number"]).copy()

            common_cols = [c for c in feat_pat.columns if c in train_df.columns]
            if common_cols:
                train_df = train_df[common_cols]
                feat_use = feat_pat[common_cols]
            else:
                train_df = feat_pat
                feat_use = feat_pat
        else:
            train_df = feat_pat
            feat_use = feat_pat

        if train_df.empty or feat_use.empty:
            continue

        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(train_df)

        preds = iso.predict(feat_use)
        anomaly_mask[df_pat.index] = preds == -1

    return anomaly_mask, None, []


def metric_snapshot(trace_df, metric_dfs, phase=None, subphase=None):
    # Create a metric snapshot for a specific service and time period.
    timestamp = datetime.now().astimezone(timezone.utc)
    
    try:
        logger.debug("metric_snapshot: starting with %d traces", 0 if trace_df is None else len(trace_df))
        response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        duration = response_times.mean() if not response_times.empty else 0
        total_requests = len(trace_df)

    except Exception as e:
        logger.error(f"Error in snapshot: {e}")
        duration = 0
        total_requests = 0

        return {"error": str(e)}
    
    # Calculate per-pattern deadline miss rates
    pattern_miss_rates = calculate_pattern_miss_rates(trace_df)
    if pattern_miss_rates:
        logger.debug("metric_snapshot: pattern_miss_rates keys: %s", list(pattern_miss_rates.keys()))
        overall_miss_rate = sum(e['miss_rate'] for e in pattern_miss_rates.values()) / len(pattern_miss_rates)
    else:
        overall_miss_rate = 0
    
    # Perform SHAP anomaly detection
    anomaly_results = perform_shap_anomaly_detection(trace_df)
    logger.info(
        "metric_snapshot: anomaly_count=%d, services_with_anomalies=%s",
        anomaly_results.get('anomaly_count', 0),
        list(anomaly_results.get('anomalies_by_service', {}).keys()),
    )
    
    # Calculate overall pattern stats using 65th percentile (same as individual patterns)
    
    # Calculate additional metrics
    p90_response_time = 0
    if not trace_df.empty and 'total' in trace_df.columns:
        response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        if not response_times.empty:
            p90_response_time = response_times.quantile(0.90)
    
    # Calculate overall CPU (cores) and memory (MB) utilization from service metrics
    overall_cpu_utilization_cores = 0
    overall_memory_utilization_mb = 0
    cpu_values = []
    memory_values = []
    
    for service_name in metric_dfs.keys():
        if service_name in metric_dfs:
            service_data = metric_dfs[service_name]
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    col_lower = col.lower()
                    if 'cpu' in col_lower:
                        cpu_val = service_data[col].mean() if not service_data[col].empty else 0
                        if cpu_val > 0:
                            cpu_values.append(cpu_val)
                    if 'memory' in col_lower:
                        mem_val = service_data[col].mean() if not service_data[col].empty else 0
                        if mem_val > 0:
                            memory_values.append(mem_val)
    
    if cpu_values:
        overall_cpu_utilization_cores = sum(cpu_values)
    if memory_values:
        overall_memory_utilization_mb = sum(memory_values) / (1024 * 1024)
    
    # Calculate anomaly rate
    anomaly_rate = (anomaly_results['anomaly_count'] / total_requests * 100) if total_requests > 0 else 0
    
    # Restructure snapshot data according to the specified format
    snapshot = {
        'timestamp': timestamp,
        'total_requests': total_requests,
        'mean_response_time': duration,
        'p90_response_time': p90_response_time,
        'deadline_miss_rate': overall_miss_rate,
        'patterns': {},
        'sum_cpu_utilization_cores': overall_cpu_utilization_cores,
        'sum_memory_utilization_mb': overall_memory_utilization_mb,
        'anomaly_count': anomaly_results['anomaly_count'],
        'anomaly_rate': anomaly_rate,
        'services': {},
        'phase': phase,
        'subphase': subphase
    }
    
    # Add patterns with the specified structure
    for pattern, stats in pattern_miss_rates.items():
        # Calculate p90 for this pattern
        pattern_p90 = 0
        if not trace_df.empty:
            pattern_traces = trace_df[trace_df['pattern'] == pattern]
            if not pattern_traces.empty and 'total' in pattern_traces.columns:
                pattern_response_times = pd.to_numeric(pattern_traces['total'], errors='coerce')
                if not pattern_response_times.empty:
                    pattern_p90 = pattern_response_times.quantile(0.90)
        
        snapshot['patterns'][pattern] = {
            'deadline_miss_rate': stats['miss_rate'],
            'mean_response_time': stats['mean_response_time'],
            'p90_response_time': pattern_p90
        }
    
    # Add service-specific data with the specified structure
    for service_name in metric_dfs.keys():
        if service_name in metric_dfs:
            service_data = metric_dfs[service_name]
            
            # Get CPU (cores) and memory (MB) utilization for this service
            service_cpu_utilization_cores = 0
            service_memory_utilization_mb = 0
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    col_lower = col.lower()
                    if 'cpu' in col_lower and service_cpu_utilization_cores == 0:
                        service_cpu_utilization_cores = service_data[col].mean() if not service_data[col].empty else 0
                    if 'memory' in col_lower and service_memory_utilization_mb == 0:
                        mem_val = service_data[col].mean() if not service_data[col].empty else 0
                        service_memory_utilization_mb = mem_val / (1024 * 1024)
            
            # Get anomaly information for this service from SHAP results
            service_anomalies = anomaly_results['anomalies_by_service'].get(service_name, [])
            service_anomaly_count = len(service_anomalies)
            service_anomaly_rate = (service_anomaly_count / total_requests * 100) if total_requests > 0 else 0
            
            snapshot['services'][service_name] = {
                'anomaly_count': service_anomaly_count,
                'anomaly_rate': service_anomaly_rate,
                'cpu_utilization_cores': service_cpu_utilization_cores,
                'memory_utilization_mb': service_memory_utilization_mb
            }
    
    return snapshot, anomaly_results


def calculate_pattern_miss_rates(trace_df):
    if trace_df is None or trace_df.empty or 'pattern' not in trace_df.columns or 'total' not in trace_df.columns:
        logger.error("Error in calculate_pattern_miss_rates: trace_df is None or empty or 'pattern' or 'total' not in trace_df.columns")
        return {}
    
    pattern_miss_rates = {}
    
    # Pre-load training traces (if available) so that per-pattern thresholds
    # can be computed from the training distribution when possible.
    training_df = _load_training_traces()

    # Group by pattern and calculate miss rates
    for pattern in trace_df['pattern'].unique():
        pattern_traces = trace_df[trace_df['pattern'] == pattern]
        response_times = pd.to_numeric(pattern_traces['total'], errors='coerce')

        # Calculate threshold as mean + 1.5 * std for this pattern,
        # preferring the training traces for this pattern when available.
        threshold = get_pattern_miss_rate_threshold(
            response_times, pattern=pattern, training_df=training_df
        )
        
        # Count requests above the threshold
        deadline_misses = len(response_times[response_times > threshold])
        total_requests = len(response_times)
        miss_rate = (deadline_misses / total_requests * 100) if total_requests > 0 else 0
        
        pattern_miss_rates[pattern] = {
            'miss_rate': miss_rate,
            'miss_count': deadline_misses,
            'total_requests': total_requests,
            'mean_response_time': response_times.mean(),
            'max_response_time': response_times.max()
        }
    
    return pattern_miss_rates


def perform_shap_anomaly_detection(trace_df):

    trace_df.reset_index()
    if trace_df is None or trace_df.empty:
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }
    
    try:
        import numpy as np
        import heapq
        from operator import itemgetter
        from config import SPAN_PROCESS_MAP
        
        # Prepare features for anomaly detection (exclude non-numeric columns)
        feature_columns = trace_df.select_dtypes(include=["number"]).columns.tolist()
        logger.debug("perform_shap_anomaly_detection: initial numeric feature columns: %s", feature_columns)

        if not feature_columns:
            return {
                'anomaly_count': 0,
                'anomalies_by_service': {},
                'shap_contributions': {}
            }

        # Load training traces once so we can build per-pattern models from
        # the corresponding training distribution.
        training_df = _load_training_traces()
        if "pattern" in training_df.columns:
            logger.debug(
                "perform_shap_anomaly_detection: training patterns (sample up to 20): %s",
                training_df["pattern"].unique()[:20],
            )

        service_anomaly_count = {}
        anomalies_by_service = {}
        shap_contributions = {}

        # Build anomalies and SHAP explanations per pattern, then aggregate.
        if 'pattern' in trace_df.columns:
            patterns = trace_df['pattern'].unique()
        else:
            patterns = [None]
        logger.debug("perform_shap_anomaly_detection: patterns found: %s", patterns)

        for pattern in patterns:
            if pattern is None:
                df_pat = trace_df
            else:
                df_pat = trace_df[trace_df['pattern'] == pattern]

            if df_pat.empty:
                logger.debug("perform_shap_anomaly_detection: pattern %s has empty df_pat; skipping", pattern)
                continue

            logger.debug(
                "perform_shap_anomaly_detection: pattern %s has %d traces before IF",
                pattern, len(df_pat)
            )

            features_pat = df_pat[feature_columns].copy()

            # Determine training data for this pattern.
            train_pat = training_df.copy()
            if 'pattern' in training_df.columns and pattern is not None:
                filtered = training_df[training_df['pattern'] == pattern]
                if not filtered.empty:
                    train_pat = filtered
                    logger.debug(
                        "perform_shap_anomaly_detection: using %d training traces for pattern %s",
                        len(train_pat),
                        pattern,
                    )
                else:
                    # No training traces for this pattern; fall back to using
                    # the current pattern's traces as the training distribution.
                    logger.debug(
                        "perform_shap_anomaly_detection: no training traces for pattern %s; "
                        "using current traces as training data",
                        pattern,
                    )
                    train_pat = df_pat

            train_pat = train_pat.select_dtypes(include=["number"]).copy()
            features_pat = features_pat.select_dtypes(include=["number"]).copy()

            # Align numeric feature columns between training and current data.
            common_cols = [c for c in features_pat.columns if c in train_pat.columns]
            if common_cols:
                train_use = train_pat[common_cols]
                features_use = features_pat[common_cols]
            else:
                # Fall back to using only the current pattern's features.
                train_use = features_pat
                features_use = features_pat

            if train_use.empty or features_use.empty:
                logger.debug(
                    "perform_shap_anomaly_detection: pattern %s has empty train/features after alignment; skipping",
                    pattern,
                )
                continue

            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            iso_forest.fit(train_use)

            anomaly_predictions = iso_forest.predict(features_use)

            # Find anomalies within this pattern
            anomaly_mask_pat = anomaly_predictions == -1

            # Filter anomalies based on the same per-pattern deadline threshold
            # used for deadline miss rate calculations.
            if "total" in df_pat.columns:
                response_times_pat = pd.to_numeric(df_pat["total"], errors="coerce")
                threshold_pat = get_pattern_miss_rate_threshold(
                    response_times_pat,
                    pattern=pattern,
                    training_df=training_df,
                )
                # Use numpy coercion to avoid type-checker confusion around `.to_numpy()`.
                time_above_threshold_mask = np.asarray(response_times_pat) > threshold_pat
            else:
                # If total is missing, keep all anomalies (can't evaluate "above threshold").
                threshold_pat = 0.0
                time_above_threshold_mask = np.ones(len(df_pat), dtype=bool)

            filtered_anomaly_mask = anomaly_mask_pat & time_above_threshold_mask

            num_anom = int(anomaly_mask_pat.sum())
            num_filtered = int(filtered_anomaly_mask.sum())
            logger.info(
                "perform_shap_anomaly_detection: pattern %s => %d anomalies after deadline filter (from %d)",
                pattern,
                num_filtered,
                num_anom,
            )

            if not filtered_anomaly_mask.any():
                continue

            anomaly_indices = df_pat.index[filtered_anomaly_mask].tolist()

            # Use the same SHAP calculation method as get_kpi_list
            # Use label-based indexing to avoid positional out-of-bounds errors
            anom_features = features_use.loc[filtered_anomaly_mask]
            shapes, names = shap_decisions(iso_forest, anom_features)

            dropped_no_service = 0

            for s, ai in zip(shapes, anomaly_indices):
                duration = pd.to_timedelta(trace_df.loc[ai, "total"], unit="us")
                timestamp = trace_df.loc[ai, "startTime"]
                
                # Get the 2 most negative SHAP values (same as get_kpi_list)
                values = heapq.nsmallest(2, enumerate(s), key=itemgetter(1))
                
                # Find the primary service using the same logic as get_kpi_list
                primary_service = None
                top_contributors = []
                
                for v in values:
                    feature_name = names[v[0]]
                    shap_value = v[1]
                    
                    # Skip certain features (same as get_kpi_list)
                    if feature_name in ['mongo_rate']:
                        continue

                    # Map feature to service using SPAN_PROCESS_MAP and os.path.basename (same as get_kpi_list)
                    if feature_name in SPAN_PROCESS_MAP:
                        service_name = os.path.basename(SPAN_PROCESS_MAP[feature_name])
                        if primary_service is None:  # Use the first valid service as primary
                            primary_service = service_name
                        
                        top_contributors.append({'feature': feature_name, 'shap_value': float(shap_value)})
                        
                        # Count anomalies per service (same as get_kpi_list)
                        service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1
                    
                    break  # Only process the first valid contributor (same as get_kpi_list)
                
                if primary_service is None:
                    dropped_no_service += 1
                    continue
                
                # Store anomaly information 
                anomaly_info = {
                    'index': int(ai),
                    'trace_id': trace_df.loc[ai, 'id'] if 'id' in trace_df.columns else None,
                    'pattern': trace_df.loc[ai, 'pattern'] if 'pattern' in trace_df.columns else None,
                    'response_time': trace_df.loc[ai, 'total'] if 'total' in trace_df.columns else None,
                    'timestamp': timestamp,
                    'duration_seconds': duration.total_seconds(),
                    'service': primary_service
                }
                
                # Group anomalies by service
                if primary_service not in anomalies_by_service:
                    anomalies_by_service[primary_service] = []
                anomalies_by_service[primary_service].append(anomaly_info)
                
                # Aggregate contributions for overall analysis
                for contrib in top_contributors:
                    feature_name = contrib['feature']
                    shap_value = contrib['shap_value']
                    if feature_name not in shap_contributions:
                        shap_contributions[feature_name] = {'count': 0, 'total_contribution': 0.0}
                    shap_contributions[feature_name]['count'] += 1
                    shap_contributions[feature_name]['total_contribution'] += abs(shap_value)
        
        # Calculate average contributions
        for feature_name in shap_contributions:
            contrib = shap_contributions[feature_name]
            contrib['average_contribution'] = contrib['total_contribution'] / contrib['count']
        
        # Sort anomalies within each service by response time (highest first)
        for service in anomalies_by_service:
            anomalies_by_service[service].sort(key=lambda x: x['response_time'], reverse=True)
        
        # Calculate total anomaly count
        total_anomaly_count = sum(service_anomaly_count.values())
        logger.info(
            "perform_shap_anomaly_detection: total_anomaly_count=%d, services=%s",
            total_anomaly_count,
            list(anomalies_by_service.keys()),
        )
        
        return {
            'anomaly_count': total_anomaly_count,
            'anomalies_by_service': anomalies_by_service,
            'shap_contributions': shap_contributions,
            'service_anomaly_counts': service_anomaly_count
        }
        
    except Exception as e:
        # print(e)
        print(traceback.format_exc())
        logger.error(f"Error in SHAP anomaly detection: {e}")
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }


def get_trace_IsolationForest() -> IsolationForest:
    """
    Backwards-compatible helper that returns a single IsolationForest trained
    on all training traces, using numeric, non-'total' columns.

    Newer code prefers to build per-pattern IsolationForest models directly
    from `_load_training_traces()`, but this is kept for existing callers.
    """
    global iso_forest, iso_forest_features
    if iso_forest is not None:
        return iso_forest, iso_forest_features

    trace_df = _load_training_traces()
    feature_columns = trace_df.select_dtypes(include=["number"]).columns.tolist()

    if 'total' in feature_columns:
        feature_columns.remove('total')

    iso_forest_features = feature_columns
    features = trace_df[feature_columns]
    _iso_forest = IsolationForest(contamination=0.01, random_state=42)
    _iso_forest.fit(features)

    iso_forest = _iso_forest

    return iso_forest, iso_forest_features


def calculate_CVaR(data, alpha=0.95):
    series = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    if series.empty:
        return 0.0
    arr = series.to_numpy(dtype=float, copy=False)
    var_alpha = np.quantile(arr, alpha)
    tail = arr[arr >= var_alpha]
    return float(np.mean(tail))


def get_pattern_miss_rate_threshold(response_times, pattern=None, training_df=None):
    """
    Deadline threshold = CVaR (expected shortfall) of response times at alpha=0.95.

    If a training traces dataframe is provided, prefer computing the
    threshold from that distribution (optionally restricted to the
    same pattern), falling back to the provided response_times.
    """
    if training_df is not None and not training_df.empty and "total" in training_df.columns:
        df = training_df
        if pattern is not None:
            if "pattern" not in df.columns:
                span_cols = df.columns.difference(["id"])
                df = df.copy()
                df["pattern"] = (
                    df[span_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)
                )
            df = df[df["pattern"] == pattern]
        series = pd.to_numeric(df["total"], errors="coerce")
    else:
        series = pd.to_numeric(response_times, errors="coerce")

    if series is None or series.empty:
        return 0.0

    cvar_alpha = calculate_CVaR(series)
    return cvar_alpha
