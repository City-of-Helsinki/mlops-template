from __future__ import annotations
from typing import Iterable, Type, Union
from numbers import Number
from parser import ParserError
from prometheus_client import Summary, Counter, Gauge, Enum, Info
import datetime as dt
import re

import numpy as np
import pandas as pd

# For some metrics, we may want to keep a FIFO que for
# calculating summary statistics over fixed number of records
# instead of a period of time that is the standard for Prometheus:


def is_timedelta_dtype(dtype) -> bool:
    """
    check if datatype is timedelta or period
    """
    return np.any(
        [
            dtype is tp
            for tp in [
                dt.timedelta,
                pd.Timedelta,
                pd.Period,
                np.timedelta64,
            ]
        ]
    )


def is_timestamp_dtype(dtype) -> bool:
    """
    check if given datatype is a timestamp
    """
    return np.any(
        [
            dtype is tp
            for tp in [dt.date, dt.time, dt.datetime, pd.Timestamp, np.datetime64]
        ]
    )


def is_time_dtype(dtype) -> bool:
    """
    check if given data type is of any time format
    """
    return is_timedelta_dtype(dtype) or is_timestamp_dtype(dtype)


def is_numeric_dtype(dtype):
    """
    check if given datatype is numeric.
    isinstance numbers.Number does not work with pandas or numpy
    """
    try:
        dtype(1) / 1  # if numeric, should not crash
        return True
    except:
        return False


def is_bool_dtype(dtype):
    """
    Check if data type is boolean
    """
    return np.any([dtype is tp for tp in [bool, np.bool_, pd.BooleanDtype]])


def is_str_dtype(dtype):
    """
    check if given datatype is string
    """
    return np.any([dtype is tp for tp in [str, np.string_, pd.StringDtype]])


def is_categorical_dtype(dtype):
    return dtype is pd.CategoricalDtype


class FifoOverwriteDataFrame:
    """
    A FIFO Queue for storing maxsize latest items.

    Properties:
     - if full, will replace the oldest value with the newest
     - can only be emptied if completely full (flush)
     - optionally clear (drop all) at flush
    """

    def __init__(
        self, columns: dict, maxsize: int = 1000, clear_at_flush: bool = False
    ):
        self.is_full = False
        self.maxsize = maxsize
        self.columns = columns
        self.clear_at_flush = clear_at_flush
        self.df = pd.DataFrame(columns=columns)

    def put(
        self, rows: Union[np.ndarray, Iterable, dict, pd.DataFrame]
    ) -> FifoOverwriteDataFrame:
        """
        Put new items to queue. If full, overwrite the oldest value.
        Return reference to self.
        """
        y_size = self.df.shape[0]
        new_data = pd.DataFrame(rows, columns=self.columns)
        if new_data.shape[0] >= self.maxsize:
            new_data = new_data.iloc[-self.maxsize :]
        self.df = pd.concat(
            (self.df.iloc[1:] if y_size == self.maxsize else self.df, new_data),
            ignore_index=True,
        )
        if self.df.shape[0] == self.maxsize:
            self.is_full = True
        else:
            self.is_full = False

        return self

    def flush(self) -> pd.DataFrame:
        """
        If queue df is full, return copy.
        If self.clear_at_flush, clear df before returning copy. Queue is only cleared if full.
        Else, return false.
        """
        if not self.is_full:
            return False
        ret = self.df.copy()
        if self.clear_at_flush:
            self.df.drop(self.df.index, inplace=True)
        return ret


# Prometheus naming conventions:
#
# suffix describing base unit, plural form
# _total for unitless count
# _unit_total for unit accumulating count
# _info for metadata
# _timestamp_seconds for timestamps
# rule of thumb: either sum() or avg() of metric must be meaningful, or metric has to be split up
#
# commonly used: (elsewhere -> in prometheus)
# time -> seconds
# percent -> ratio  0-1
# bits, bytes -> bytes


def convert_time_to_seconds(
    t,
    errors: str = "raise",
    pd_infer_datetime_format: bool = True,
    pd_str_parse_format: bool = None,
):
    """
    Prometheus standard is to express all time in seconds.
    This function parses time expressions to seconds in accuracy of floats.

    Recursive function, parses:
        strings -> pandas -> numpy -> float
        datetime -> float
        int -> float

    Parameters:

        errors: {‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
            - If 'raise', raise an exception.
            - If 'coerce', return np.nan.
            - If 'ignore', return the input.

        pd_infer_datetime_format: bool, default True

        pd_str_parse_format: str, default none

    By default try to infer format from string inputs. This is slower compared to parsing defined format.
    Overwrite by passing custom format to parameter pd_str_parse_format.
    Force custom format by setting pd_infer_datetime_format to False.
    """

    if errors not in ["raise", "ignore", "coerce"]:
        raise ValueError(f"{errors} is not a valid argument for parameter errors!")

    try:
        # strings
        if is_str_dtype(type(t)):
            try:
                try:
                    ret = pd.to_datetime(
                        t,
                        infer_datetime_format=pd_infer_datetime_format,
                        format=pd_str_parse_format,
                    )
                except (pd.errors.ParserError, ValueError):
                    try:
                        ret = pd.to_timedelta(t)
                    except pd.errors.ParserError:
                        ret = pd.Period(t)
            except ValueError:
                raise ValueError(
                    f"Unsupported expression of time: {t}"
                    + "\nDo you have a metric with name *time* or *date* that is not an expression of time?"
                )
            return convert_time_to_seconds(ret)

        # pandas
        elif isinstance(t, (pd.Timestamp, pd.Timedelta)):
            return convert_time_to_seconds(t.to_numpy())
        elif isinstance(t, pd.Period):
            return convert_time_to_seconds(
                t.to_timestamp(how="E") - t.to_timestamp(how="S")
            )

        # numpy
        elif isinstance(t, np.datetime64):
            return convert_time_to_seconds(t - np.datetime64("1970-01-01"))
        elif isinstance(t, np.timedelta64):
            return t / np.timedelta64(1, "s")

        # datetime
        elif isinstance(t, dt.date):
            return float(((t.year * 12 + t.month) * 31 + t.day) * 24 * 60 * 60)
        elif isinstance(t, dt.time):
            return float((t.hour * 60 + t.minute) * 60 + t.second)
        elif isinstance(t, dt.datetime):
            return convert_time_to_seconds((t - dt.datetime.min).timestamp())
        elif isinstance(t, dt.timedelta):
            return t / dt.timedelta(seconds=1)
        # other (int, float)
        else:
            return float(t)
    except Exception as e:
        if errors == "raise":
            raise e
        elif errors == "coerce":
            return np.nan
        else:
            return t


def create_promql_metric_name(
    metric_name: str,
    dtype: Type = None,
    prefix: str = "",
    suffix: str = "",
    is_counter=False,
):
    """
    Create a promql compatible name for a metric.
    Note that this does not perfectly ensure good naming,
    but may help when auto-generating metrics.

    See https://prometheus.io/docs/practices/naming/ for naming conventions.
    """

    ret = prefix + "_" + metric_name + "_" + suffix
    # must be lowercase
    ret = ret.lower().strip("_")

    if is_timestamp_dtype(dtype):
        # e.g. date_of_birth -> 'date_of_birth_timestamp_seconds'
        ret = ret.replace("seconds", "").replace("timestamp", "") + "_timestamp_seconds"

    elif is_timedelta_dtype(dtype):
        # e.g. time_on_site -> 'time_on_site_seconds'
        ret = ret.replace("seconds", "") + "_seconds"
    elif is_str_dtype(dtype):
        # e.g. description -> description_count
        ret += "_info"
    else:  # add more exceptions if needed
        pass

    # remove metric types from metric name (a metric name should not contain these)
    for metric_type in ["gauge", "counter", "summary", "map"]:
        ret = ret.replace("_" + metric_type + "_", "")
        while ret.startswith(metric_type + "_"):
            ret = ret[: -(len(metric_type) + 1)]
        while ret.endswith("_" + metric_type):
            ret = ret[: -(len(metric_type) + 1)]

    ret = ret.strip("_")

    # remove reserved suffixes (a metric should not end with these)
    for reserved_suffix in ["_count", "_sum", "_bucket", "_total"]:
        l = len(reserved_suffix)
        while ret.endswith(reserved_suffix):
            ret = ret[:-l]

    # however, counters should always have _total suffix
    if is_counter:
        ret += "_total"
    # clean non-alphanumericals except underscores
    ret = re.sub(r"[^\w" + "_" + "]", "_", ret)

    # clean extra underscores
    ret = "".join(
        ["_" + val.strip("_") if val != "" else "" for val in ret.split("_")]
    ).strip("_")

    return ret


def record_metrics_from_dict(
    metrics: dict, convert_names_to_promql: bool = True
) -> list:
    """
    Read pre-recorded metrics from dict to Prometheus.
    Use case: pass model train/val pipeline metrics to prometheus.

    This allows recording three types of metrics
        - numeric (int, float, etc.)
        - categorical
        - info (metadata)
    Lists and matrices must be split so that each cell is their own metric.

    For designing metrics, see Prometheus naming conventions see:
        https://prometheus.io/docs/practices/naming/
    For metadata (model version, data version etc. see:
        https://www.robustperception.io/exposing-the-software-version-to-prometheus/

    Format:
    metrics = {
        'metric_name':{
            'value': int, float or str if 'type' = category,
            'description': str,
            'type': str -> 'numeric', 'category' or 'info' (metadata / pseudo metrics),
            'categories': [str], e.g. ['A', 'B', 'C']. only required if 'type' = category,
            'label_names': [str], optional. for info type metrics
            'label_values': [str], optional. for info type metrics
        }
    }

    Example:
    metrics = {
        "train_loss": {
            "value": 0.95,
            "description": "training loss (MSE)",
            "type": "numeric",
        },
        "test_loss": {
            "value": 0.96,
            "description": "test loss (MSE)",
            "type": "numeric",
        },
        "optimizer": {
            "value": np.random.choice(["SGD", "RMSProp", "Adagrad"]),
            "description": "ml model optimizer function",
            "type": "category",
            "categories": ["SGD", "RMSProp", "Adagrad", "Adam"],
        },
        "model_build_info": {
            "description": "dev information",
            "type": "info",
            "value": {'origin':'city-of-helsinki@github.com/ml-app', 'branch':'main', 'commit':'12345678'}
        },
    }

    """
    ret = []  # return metric handles. In normal use these are not needed.

    for metric_name in metrics.keys():
        metric = metrics[metric_name]
        dtype = type(metric["value"])
        # prometheus name for metric
        metric_handle = (
            create_promql_metric_name(metric_name=metric_name, dtype=dtype)
            if convert_names_to_promql
            else metric_name
        )

        value = metric["value"]
        # convert time formats to seconds
        value = (
            convert_time_to_seconds(value)
            if is_time_dtype(type(value))
            or metric_handle.endswith(("seconds", "timestamp"))
            else metric["value"]
        )

        # TODO: consider inferring types from dtype instead of user definition
        if metric["type"] == "numeric":
            m = Gauge(metric_handle, metric["description"])
            m.set(value)
        elif metric["type"] == "category":
            m = Enum(metric_handle, metric["description"], states=metric["categories"])
            m.state(value)
        elif metric["type"] == "info":
            m = Info(
                metric_handle,
                metric["description"],
            )
            # WARNING: each new metric-label combination creates a new time series!
            m.info(value)
        else:
            raise ValueError(f"metric of unknown type: {metric}")
        ret.append(m)

    return ret


# TODO: PROMEHEUS:
# input:
#   - raw values (if not text or some other weird datatype)
#   - hist/sumstat (a bit more private)

# TODO: unittest
class SummaryStatisticsMetrics:
    """
    Class wrapper for generic drift monitoring.

    Generate and update prometheus metrics based on a dataframe,
    using a summary statistics function.

    By default, uses pd.DataFrame.describe(), but custom summary statistics may be used.

    Designed to be used to calculate summary statistics of input data collected to
    FifoOverwriteDataFrame and pass them to prometheus.

    Workflow example:

    0. SummaryStatisticsMetrics creates prometheus metrics for drift detection when api is launched.
    1. Input features from requests to api collected to FifoOverwriteDataFrame.
    2.0 FODF flushed when full or periodically.
    2.1 (optional) Flush data can be anonymized if needed, for example with Helsinki tabular anonymizer.
    3. SummaryStatisticsMetrics calculates summary statistics from flush and updates them to prometheus.
    4. Return to step 1.
    """

    def init(
        self,
        columns: dict,
        summary_statistics_function: function = lambda df: df.describe(include="all"),
        convert_names_to_promql: bool = True,
        metrics_name_prefix: str = "",
    ):
        """
        Parameters:

        columns: dict of column-datatype pairs similar to when creating pd.DataFrame
        summary_statistics_function: function that takes in pd.DataFrame returns
            a dataframe containing summary statistics for each column.
        convert_names_to_promql: metric names are inferred from given column and summary statistic function.
            If true, inferred names are auto-corrected to promql.
        metrics_name_prefix: an optional prefix to prometheus metric names created, e.g. 'input_'
        """
        self.columns = columns
        self.summary_statistics_function = summary_statistics_function

        # initialize metric names by calling summary statistics on an empty dataframe
        self.rownames = summary_statistics_function(
            pd.DataFrame(columns=dict)
        ).index.values
        self.colnames = list(self.columns.keys())

        self.metrics = {}  # store metric handles in a dict
        for colname in self.colnames:
            dtype = columns["colname"]["dtype"]
            for rowname in self.rownames:
                metric_key = "_".join([colname, rowname])
                metric_description = f"calculated using summary statistics function {summary_statistics_function.__name__}"
                metric_name = (
                    create_promql_metric_name(
                        metric_name=metric_key, dtype=dtype, prefix=metrics_name_prefix
                    )
                    if convert_names_to_promql
                    else metric_name
                )
                # if category, create Enum
                if (
                    is_time_dtype(dtype)
                    or is_numeric_dtype(dtype)
                    or is_bool_dtype(dtype)
                ):
                    g = Gauge(metric_name, metric_description)
                else:  # string, categories & objects
                    g = Enum(metric_name, metric_description, states=[])

                # record created metric in a dict
                self.metrics[metric_key] = g

    def set(self, df: pd.DataFrame):
        """
        Calculate summary statistics and set metrics accordingly
        """
        sumstat_df = self.summary_statistics_function(df)
        # loop through dataframe and calculate summary statistics
        for colname in self.colnames:
            for rowname in self.rownames:
                metric_key = "_".join([colname, rowname])
                metric_value = sumstat_df.loc[rowname, colname]
                dtype = type(metric_value)
                if is_time_dtype(dtype):  # convert all time formats to integer seconds
                    metric_value = convert_time_to_seconds(
                        metric_value, type(metric_value)
                    )
                    self.metrics[metric_key].set(metric_value)
                elif is_bool_dtype(dtype):  # convert boolean to 1-0
                    metric_value = 1 if metric_value else 0
                    self.metrics[metric_key].set(metric_value)
                elif is_numeric_dtype(dtype):  # numeric pass as is
                    self.metrics[metric_key].set(metric_value)
                elif dtype is pd.CategoricalDtype:  # categoricals -> enum
                    self.metrics[metric_key].state(metric_value)
                elif metric_value is None or np.isnan(
                    metric_value
                ):  # do not record nans
                    pass
                else:
                    try:  # other values should be convertable to string
                        metric_value = str(metric_value)
                        self.metrics[metric_key].info(metric_value)
                    except:
                        pass  # do not record non-numeric, boolean, categorical or non-convertable to str


# processing:
#   - time (total / hist )
#   - general resource usage
#   - request counter
# output:
#   - raw (if not text of some other weird datatype)
#   - if category
#   - hist/sumstat (a bit more private)
#   - live_scoring
