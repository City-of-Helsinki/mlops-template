from __future__ import annotations
import functools
from typing import Iterable, Type, Union
from numbers import Number
import os
from prometheus_client import generate_latest, Counter, Gauge, Enum, Info
import pyarrow.feather as feather
import datetime as dt
import re
import time

from itertools import product

import numpy as np
import pandas as pd

# functions for checking data types:


def is_timedelta(dtypename) -> bool:
    """
    check if datatype is timedelta or period
    """
    return np.any(
        [
            tp in dtypename
            for tp in [
                "timedelta",
                "Timedelta",
                "period",
                "Period",
                "timedelta64",
            ]
        ]
    )


def is_timestamp(dtypename) -> bool:
    """
    check if given datatype is a timestamp
    """
    return (
        dtypename == "date"
        or dtypename == "time"
        or np.any(
            [
                tp in dtypename
                for tp in [
                    "datetime",
                    "timestamp",
                    "Timestamp",
                    "datetime64",
                ]
            ]
        )
    )


def is_time(dtypename) -> bool:
    """
    check if given data type is of any time format
    """
    return is_timedelta(dtypename) or is_timestamp(dtypename)


def is_numeric(dtypename) -> bool:
    """
    check if given datatype is numeric.
    isinstance numbers.Number does not work with pandas or numpy
    """
    return np.any(
        [
            dtypename == "".join(x)
            for x in product(
                ["int", "uint", "float"], ["", "_", "8", "16", "32", "64", "128"]
            )
        ]
    )


def is_bool(dtypename) -> bool:
    """
    Check if data type is boolean
    """
    return np.any(
        [dtypename == tp for tp in ["bool", "bool_", "boolean", "BooleanDtype"]]
    )


def is_str(dtypename) -> bool:
    """
    check if given datatype is string
    """
    return np.any(
        [
            dtypename == tp
            for tp in [
                "<U11",
                "str",
                "str_",
                "string",
                "string_",
                "object",
                "StringDtype",
            ]
        ]
    )


# NOTE: CATEGORIES, pd.Categorical can not be extracted outside pandas:
# it is always converted back to it's original form
# therefore we can not check from a single value wheather or not it is categorical:
# it must be checked from the dataframe itself on a higher level
def is_categorical(dtypename) -> bool:
    raise NotImplementedError(
        "Categorical values are typecasted back to their original dtype outside pandas.\nCheck categorical columns straight from the dataframe instead."
    )


def is_object(dtypename) -> bool:
    """
    if nothing else, treat as an generic object data type
    """
    return not (
        is_time(dtypename)
        or is_numeric(dtypename)
        or is_bool(dtypename)
        or is_str(dtypename)
    )


def get_dtypename(dtype) -> str:
    """
    return datatype name for given datatype
    """
    try:
        return dtype.__name__
    except AttributeError:
        return str(dtype)


def value_dtypename(value) -> str:
    """
    return datatype name for given value
    """
    try:
        return get_dtypename(value.dtype)
    except AttributeError:
        return get_dtypename(type(value))


# convert time expressions to seconds (prometheus convention)


def convert_time_to_seconds(
    value,
    errors: str = "raise",
    pd_infer_datetime_format: bool = True,
    pd_str_parse_format: bool = None,
) -> float:
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

    dtypename = value_dtypename(value)
    try:
        # strings
        if is_str(dtypename):
            try:
                try:
                    ret = pd.to_datetime(
                        value,
                        infer_datetime_format=pd_infer_datetime_format,
                        format=pd_str_parse_format,
                    )
                except (pd.errors.ParserError, ValueError):
                    try:
                        ret = pd.to_timedelta(value)
                    except pd.errors.ParserError:
                        ret = pd.Period(value)
            except ValueError:
                raise ValueError(
                    f"Unsupported expression of time: {value}"
                    + "\nDo you have a metric with name *time* or *date* that is not an expression of time?"
                )
            return convert_time_to_seconds(ret)

        # pandas
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return convert_time_to_seconds(value.to_numpy())
        elif isinstance(value, pd.Period):
            return convert_time_to_seconds(
                value.to_timestamp(how="E") - value.to_timestamp(how="S")
            )

        # numpy
        elif isinstance(value, np.datetime64):
            return convert_time_to_seconds(value - np.datetime64("1970-01-01"))
        elif isinstance(value, np.timedelta64):
            return float(value / np.timedelta64(1, "s"))

        # datetime
        elif dtypename == "date":  # isinstance catches non-dates too!
            return float(
                ((value.year * 12 + value.month) * 31 + value.day) * 24 * 60 * 60
            )
        elif dtypename == "time":  # isinstance catches non-times too!
            return float((value.hour * 60 + value.minute) * 60 + value.second)
        elif isinstance(value, dt.datetime):
            return convert_time_to_seconds((value - dt.datetime.min))
        elif isinstance(value, dt.timedelta):
            return value / dt.timedelta(seconds=1)
        # other (int, float)
        else:
            return float(value)
    except Exception as e:
        if errors == "raise":
            raise e
        elif errors == "coerce":
            return np.nan
        else:
            return value


def string_is_time(s: str) -> bool:
    """
    Check if given string is a valid time expression
    """
    try:
        convert_time_to_seconds(s)
        return True
    except ValueError as e:
        return False


# convert names to promql


def convert_metric_name_to_promql(
    metric_name: str,
    dtype: Type = None,
    dtypename: str = None,
    prefix: str = "",
    suffix: str = "",
    is_counter=False,
    mask_type_aliases=True,
    type_mask="typemask",
    mask_reserved_suffixes=True,
    suffix_mask="suffixmask",
    category=False,
) -> str:
    """
    Create a promql compatible name for a metric.
    Note that this does not perfectly ensure good naming,
    but may help when auto-generating metrics.

    See https://prometheus.io/docs/practices/naming/ for naming conventions.

    Prometheus naming conventions in a nutshell:
    - suffix describing base unit, plural form
    - _total for unitless count
    - _unit_total for unit accumulating count
    - _info for metadata
    - _timestamp_seconds for timestamps
    - rule of thumb: either sum() or avg() of metric must be meaningful, or metric has to be split up
    - time -> seconds
    - percent -> ratio  0-1
    - bits, bytes -> bytes
    """

    if dtypename is None:
        if dtype is not None:
            dtypename = get_dtypename(dtype)
        else:
            raise ValueError(
                "convert_metric_name_to_promql must be given non-None dtype or dtypename"
            )

    # add possible prefix & suffix to metric name
    ret = prefix + "_" + metric_name + "_" + suffix
    # must be lowercase
    ret = ret.lower().strip("_")

    # can't begin with non-alphabetical
    if not ret[:1].isalpha() and ret[:1].isnumeric():
        ret = (
            "column" + ret
        )  # assuming this would rise from unnamed columns addressed by index

    if is_timestamp(dtypename):
        # e.g. date_of_birth -> 'date_of_birth_timestamp_seconds'
        ret = ret.replace("seconds", "").replace("timestamp", "") + "_timestamp_seconds"

    elif is_timedelta(dtypename):
        # e.g. time_on_site -> 'time_on_site_seconds'
        ret = ret.replace("seconds", "") + "_seconds"

    else:  # add more exceptions if needed
        # info suffix is inferred automatically if metric type is info
        # TODO: percent -> ratio  0-1 (if possible)
        # TODO: bits, bytes -> bytes (if possible)
        pass

    # remove metric types from metric name (a metric name should not contain these)
    for metric_type in ["gauge", "counter", "summary", "map"]:
        if not mask_type_aliases:  # remove
            ret = ret.replace(metric_type, "")
        else:  # remove by masking
            ret = ret.replace(metric_type, metric_type + type_mask)

    ret = ret.strip("_")

    # remove reserved suffixes (a metric should not end with these)
    if not is_counter:
        for reserved_suffix in ["_count", "_sum", "_bucket", "_total"]:
            if not mask_reserved_suffixes:  # remove
                l = len(reserved_suffix)
                while ret.endswith(reserved_suffix):
                    ret = ret[:-l]
            elif ret.endswith(reserved_suffix):  # remove by masking
                ret += suffix_mask
    elif not ret.endswith("_total"):  # however, counters should always end with _total
        ret += "_total"

    # clean non-alphanumericals except underscores
    ret = re.sub(r"[^\w" + "_" + "]", "_", ret)

    # clean extra underscores
    ret = "".join(
        ["_" + val.strip("_") if val != "" else "" for val in ret.split("_")]
    ).strip("_")

    # return promql compatible metric name
    return ret


def record_metrics_from_dict(
    metrics: dict, convert_names_to_promql: bool = True
) -> list:
    """
    Read pre-recorded metrics from dict to Prometheus.

    Use case: pass model train/val pipeline metrics to prometheus.

    This allows recording three types of metrics
        - numeric (gauge) (int, float, etc.)
        - categorical (enum)
        - string (info), metadata
    Lists and matrices must be split so that each cell is their own metric.

    For metadata (model version, data version etc. see:
        https://www.robustperception.io/exposing-the-software-version-to-prometheus/

    Input format:
    metrics = {
        'metric_name':{
            'value': int, float or str if 'type' = category,
            'description': str,
            'type': str -> 'numeric', 'category' or 'info' (metadata / pseudo metrics),
            'categories': [str], e.g. ['A', 'B', 'C']. only required if 'type' = category
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
    # return metric handles, even if they are not needed again.
    ret = []

    # loop through recorded values
    for metric_name in metrics.keys():
        metric = metrics[metric_name]
        dtype = type(metric["value"])
        # prometheus name for metric
        metric_handle = (
            convert_metric_name_to_promql(metric_name=metric_name, dtype=dtype)
            if convert_names_to_promql
            else metric_name
        )

        value = metric["value"]
        dtypename = value_dtypename(value)
        # convert time formats to seconds
        value = (
            convert_time_to_seconds(value)
            if is_time(dtypename) or metric_handle.endswith(("seconds", "timestamp"))
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


class DriftQueue:
    """
    A FIFO overwrite queue for storing [maxsize] latest items.
    Intented for computing drift metrics for ML input, output & model performance.

    Properties:
     - if full, will replace the oldest value with the newest
     - can only be emptied (flushed) if completely full by default
     - clear df (drop all) at non-empty flush by default. This helps to limit the cost
        of computing drift metrics, as they will only need to be calculated every [maxsize]
        datapoints
     - if given a filename for backup, will try to initialize and back up the queue to given file.
        This is to avoid data loss due to container failures etc.

    Parameters:
        columns: dict of name-type pairs to build a pd.DataFrame
        backup_file: str, a complete filepath. .feather suffix recommended. if empty, no backup used.
        maxsize: int, size of the fifo queue (dataframe rows)
        clear_at_flush: bool, if true, clear queue at flush if non-empty dataframe is returned
        only_flush_full: bool, if true, you can only get values from full queue

    """

    def __init__(
        self,
        columns: dict,
        backup_file: str = "",
        maxsize: int = 1000,
        clear_at_flush: bool = True,
        only_flush_full: bool = True,
    ):
        self.maxsize = maxsize
        self.columns = columns
        self.clear_at_flush = clear_at_flush
        self.only_flush_full = only_flush_full
        self.backup_file = backup_file
        # initialize from backup file if given one
        if backup_file != "":
            try:
                with open(backup_file, "rb") as f:
                    self.df = feather.read_feather(f)
                    self._cut_to_maxsize()
            except FileNotFoundError:
                self.df = pd.DataFrame(columns=columns)
        else:  # else just create new
            self.df = pd.DataFrame(columns=columns)

    def is_full(self) -> bool:
        """
        Check if queue has maxsize elements
        """
        if self.df.shape[0] >= self.maxsize:
            return True
        else:
            return False

    def _cut_to_maxsize(self):
        # drop oldest rows exceeding maxsize
        if self.is_full():
            self.df = self.df.iloc[-self.maxsize :]

    def put(self, rows: Union[np.ndarray, Iterable, dict, pd.DataFrame]) -> DriftQueue:
        """
        Put new items to queue. If full, overwrite the oldest value.
        Overwrite backupfile.
        Return reference to self.
        """
        # add new data
        self.df = pd.concat(
            (self.df, pd.DataFrame(rows, columns=self.columns)),
            ignore_index=True,
        )

        self._cut_to_maxsize()

        # write backup for queue
        if self.backup_file != "":
            with open(self.backup_file, "wb") as f:
                feather.write_feather(self.df, f)

        return self

    def flush(self) -> pd.DataFrame:
        """
        Return queue contents as dataframe if full, or non-full flush permitted,
        else return an empty dataframe.
        Clear queue after flushing if required.
        If empty dataframe would be returned, but queue is not empty,
        queue is not cleared to not to loose data.
        """
        ret = self.df.copy()
        # return empty dataframe if not completely full
        if self.only_flush_full and not self.is_full():
            ret.drop(ret.index, inplace=True)
        # only allow clear queue if return is not empty
        elif ret.shape[0] > 0 and self.clear_at_flush:
            self.df.drop(self.df.index, inplace=True)
        return ret


def default_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic summary statistics function, calculates large number of descriptive statistics
    """
    return df.describe(include="all", datetime_is_numeric=True).rename(
        {"count": "sample_size"}
    )


def distribution_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic distribution metrics
    """
    return df.aggregate(
        ["count", "min", "mean", pd.DataFrame.median, "std", "max"]
    ).rename({"count": "sample_size"})


def mean_max_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Narrow summary statistics function, for performance monitoring
    """
    return df.aggregate(["count", "mean", "max"]).rename({"count": "sample_size"})


def categorical_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary staticstics function for hard clustering labels
    """
    ret = pd.DataFrame()
    for colname in df.columns.values:
        counts = df[colname].value_counts(dropna=False)
        counts.index = colname + "_proportion_of_" + counts.index.values + "_rate"
        rates = counts / counts.sum()
        ret = pd.concat((ret, rates), ignore_index=False)
    ret = ret.astype(float)
    ret = pd.concat(
        (ret, pd.DataFrame([[df.shape[0]]], index=["sample_size"]).astype(int)),
        ignore_index=False,
    )
    ret.rename(columns={list(ret)[0]: "_"}, inplace=True)
    return ret


def simple_text_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple summary statistics for text data
    """

    def word_count(s: str) -> dict:
        return {"word_count": len(s.split())}

    def string_length(s: str) -> dict:
        return {"string_length": len(s)}

    def vovel_rate(s: str) -> dict:
        ret = {}
        for vovel in ["a", "e", "i", "o", "u"]:
            ret["vovel_" + vovel + "_rate"] = s.count(vovel) / len(s)
        return ret

    ret = pd.DataFrame()
    ret["sample_size"] = [df.shape[0]]
    ret = ret.T

    # apply all the summary statistics functions to each column of the df
    for f in [word_count, string_length, vovel_rate]:
        buff = df.applymap(f)
        for colname in buff.columns.values:
            buff2 = (
                buff[colname].apply(pd.Series).aggregate(["mean"])
            )  # , 'std']) # you can edit the aggregate functions
            for colname2 in buff2.columns.values:
                buff2[colname2].index = (
                    str(colname) + "_" + colname2 + "_" + buff2[colname2].index
                )
                # save results in a single df
                ret = pd.concat((ret, buff2[colname2]))

    ret.rename(columns={list(ret)[0]: "_"}, inplace=True)
    return pd.DataFrame()


class SummaryStatisticsMetrics:
    """
    Class wrapper for generic drift monitoring.

    Generate and update prometheus metrics based on a dataframe,
    using a summary statistics function.

    By default, uses pd.DataFrame.describe(), but custom summary statistics may be used.
    If custom summary statistics function is used, given empty dataframe, it must
    return empty dataframe with columns and datatypes defined matching what would be returned with an
    non-empty dataframe. Summary statistics values must be convertable to numeric
    or string or they will not be recorded.

    Designed to be used to calculate summary statistics of input data collected to
    DriftQueue and pass them to prometheus.

    Workflow example:

    0. SummaryStatisticsMetrics creates prometheus metrics for drift detection when api is launched.
    1. Input features from requests to api collected to DriftQueue.
    2.0 DriftQueue flushed when full.
    2.1 (optional) Flush data can be anonymized if needed, for example with Helsinki tabular anonymizer.
    3. SummaryStatisticsMetrics calculates summary statistics from flush and updates them to prometheus.
    4. Return to step 1.
    """

    def __init__(
        self,
        summary_statistics_function: function = default_summary_statistics,
        convert_names_to_promql: bool = True,
        metrics_name_prefix: str = "",
    ):
        """
        Parameters:

        summary_statistics_function: function that takes in pd.DataFrame returns
            a dataframe containing summary statistics for each column.
        convert_names_to_promql: metric names are inferred from given column and summary statistic function.
            If true, inferred names are auto-corrected to promql.
        metrics_name_prefix: an optional prefix to prometheus metric names created, e.g. 'input_'
        """
        self.summary_statistics_function = summary_statistics_function
        self.convert_names_to_promql = convert_names_to_promql
        self.metrics_name_prefix = metrics_name_prefix
        self.sumstat_df = pd.DataFrame()
        # Pandas category information is not conserved element-wise.
        # To ensure categorical variables are correctly
        # implemented as enum metrics,
        # force categorical columns to enum metrics instead of info strings
        # when summary statistics are calculated column-wise.
        # to do that, keep track of input df columns and dtypes
        self.input_df_columns = pd.DataFrame().columns
        self.input_df_dtypes = pd.DataFrame().dtypes
        # store metric handles in a dict
        self.metrics = {}

    def _create_metric(self, colname, rowname, dtypename, categories):
        """
        Internal: not to be called directly but by 'calculate'.
        Create new prometheus metric if it does not already exist.
        """
        metric_key = f"{colname}_{rowname}"
        metric_description = f"calculated using summary statistics function {self.summary_statistics_function.__name__}"
        # create name for metric
        metric_name = (
            convert_metric_name_to_promql(
                metric_name=metric_key,
                dtypename=dtypename,
                prefix=self.metrics_name_prefix,
                category=categories is not None,
            )
            if self.convert_names_to_promql
            else metric_name
        )
        # if category, create Enum
        if is_str(dtypename) and categories is not None:
            m = Enum(metric_name, metric_description, states=categories)
        # gauge
        elif is_time(dtypename) or is_numeric(dtypename) or is_bool(dtypename):
            m = Gauge(metric_name, metric_description)
        # string & rest
        else:
            m = Info(metric_name, metric_description)

        # store metric handle in a dict
        self.metrics[metric_key] = m

    def get_metrics(self) -> dict:
        """
        Return summary statistics metrics in a dict
        """
        return self.metrics

    def get_sumstat(self) -> pd.DataFrame:
        """
        Return copy of summary statistics in a dataframe
        """
        return self.sumstat_df.copy()

    def calculate(self, df: pd.DataFrame) -> SummaryStatisticsMetrics:
        """
        Calculate summary statistics and store to self.sumstat_df. Return self.
        Also check for categorical variables if sumstat & input columns match.
        """
        self.input_df_columns = df.columns
        self.input_df_dtypes = df.dtypes
        self.sumstat_df = self.summary_statistics_function(df)

        # check if sumstat & input share all columns
        if self.input_df_columns.values.sort() == self.sumstat_df.columns.values.sort():
            # boolean array where true indicates that the variable is categorical
            self.category_indicator = (
                np.array([get_dtypename(dt) for dt in self.input_df_dtypes])
                == "category"
            )
        else:
            self.category_indicator = np.zeros(sumstat_df.shape[1])
        self.categories_list = []  # store categories for creating enums
        for categorical, colname in zip(self.category_indicator, self.input_df_columns):
            if categorical:
                self.categories_list.append(list(df[colname].cat.categories.values))
            else:
                self.categories_list.append(None)

        return self

    def set_metrics(self) -> SummaryStatisticsMetrics:
        """
        Set metrics from sumstat_df. Create new metric if needed. Return self.
        """
        sumstat_df = self.sumstat_df
        colnames = sumstat_df.columns
        rownames = sumstat_df.index.values
        # loop through summary statistics & set metrics accordingly
        for colname, categories in zip(colnames, self.categories_list):
            for rowname in rownames:
                metric_key = f"{colname}_{rowname}"
                metric_value = sumstat_df.loc[rowname, colname]
                dtypename = value_dtypename(metric_value)
                # create new metric if needed
                if metric_key not in self.metrics.keys():
                    self._create_metric(colname, rowname, dtypename, categories)
                # record metric values and do necessary type conversions
                if is_time(dtypename):  # convert all time formats to integer seconds
                    metric_value = convert_time_to_seconds(metric_value)
                    self.metrics[metric_key].set(metric_value)
                elif is_bool(dtypename):  # convert boolean to 1-0
                    metric_value = 1 if metric_value else 0
                    self.metrics[metric_key].set(metric_value)
                elif is_numeric(dtypename):  # numeric pass as is
                    self.metrics[metric_key].set(metric_value)
                elif (
                    is_str(dtypename) and categories is not None
                ):  # categoricals -> enum
                    self.metrics[metric_key].state(metric_value)
                elif metric_value is None or pd.isnull(
                    metric_value
                ):  # do not record nans
                    pass
                else:
                    try:  # try converting to float
                        metric_value = float(metric_value)
                        self.metrics[metric_key].set(metric_value)
                    except:
                        try:
                            # other values should at least be convertable to string
                            metric_value = str(metric_value)
                            self.metrics[metric_key].info(metric_value)
                        except:
                            pass  # do not record non-numeric, boolean, categorical or non-convertable to str
        return self


class DriftMonitor(DriftQueue, SummaryStatisticsMetrics):
    """
    Wrapper for using DriftQueue and SummaryStatisticsMetrics together
    """

    def __init__(
        self,
        columns: dict,
        maxsize: int = 1000,
        backup_file: str = "",
        clear_at_flush: bool = True,
        only_flush_full: bool = True,
        summary_statistics_function: function = default_summary_statistics,
        convert_names_to_promql: bool = True,
        metrics_name_prefix: str = "",
    ):

        # init base classes
        DriftQueue.__init__(
            self,
            columns=columns,
            maxsize=maxsize,
            backup_file=backup_file,
            clear_at_flush=clear_at_flush,
            only_flush_full=only_flush_full,
        )
        SummaryStatisticsMetrics.__init__(
            self,
            summary_statistics_function=summary_statistics_function,
            convert_names_to_promql=convert_names_to_promql,
            metrics_name_prefix=metrics_name_prefix,
        )

    def update_metrics(self) -> DriftMonitor:
        """
        If enough new data, calculate new sumstat and updates prometheus metrics accordingly.
        """
        latest_input = self.flush()
        if not latest_input.empty:
            self.calculate(latest_input).set_metrics()
        return self

    def update_metrics_decorator(self):
        """
        Use update_metrics as decorator
        """

        def wrapper1(function):
            @functools.wraps(function)
            def wrapper2(*args, **kwargs):
                self.update_metrics()
                return function(*args, **kwargs)

            return wrapper2

        return wrapper1


# util & wrappers


class RequestMonitor(DriftMonitor):
    """
    DriftMonitor wrapper for monitoring request & processing times
    """

    def __init__(self, maxsize: int = 1000):
        super().__init__(
            columns={
                "processing_time_seconds": float,
                "size_rows": int,
                "mean_by_row_processing_time_seconds": float,
            },
            backup_file="processing_fifo.feather",
            metrics_name_prefix="predict_request_",
            summary_statistics_function=mean_max_summary_statistics,
            maxsize=maxsize,
        )

        self.request_counter = Counter(
            "predict_requests", "How many requests have been received in total?"
        )
        self.prediction_counter = Counter(
            "predict_request_predictions",
            "How many individual predictions have been made in total? ",
        )

    def monitor(self):
        """
        Decorator. Count requests, predictions and time it takes to process a request & predictions
        """

        def timer(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                # add to requrest & prediction counters
                self.request_counter.inc()
                self.prediction_counter.inc(len(kwargs["p_list"]))
                # time response
                start = time.time()
                response = function(*args, **kwargs)
                end = time.time()
                processing_time = end - start
                N = len(response) + 1  # how many rows in request
                self.put([[processing_time, N, processing_time / N]])
                #
                return response

            return wrapper

        return timer


def pass_api_version_to_prometheus():
    """
    Pass git branch & HEAD for prometheus.
    Return info metric handle.
    """
    m = Info("api_git_version", "The branch and HEAD commit the api was built on.")
    m.info({"branch": os.environ["GIT_BRANCH"], "head": os.environ["GIT_HEAD"]})
    return m


def generate_metrics():
    """
    Wrapper for prometheus_client generate_latest()
    """
    return generate_latest()


def monitor_input(driftmonitor: DriftMonitor):
    """
    Monitor inputs of requests: summary statistics, count requests and individual rows in all requests
    """

    def monitor(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # loop through parameters
            input_values = []
            for p in kwargs["p_list"]:
                parameter_array = [getattr(p, k) for k in vars(p)]
                input_values.append(parameter_array)
            # update driftmonitor
            driftmonitor.put(input_values)
            # call function with parameters
            return function(*args, **kwargs)

        return wrapper

    return monitor


def monitor_output(driftmonitor: DriftMonitor):
    """
    Monitor outputs of requests: summary statistics
    """

    def monitor(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # call function
            ret = function(*args, **kwargs)
            # loop through response and extract values
            output_values = []
            for p in ret:
                label_array = [getattr(p, k) for k in vars(p)]
                output_values.append(label_array)
            # update DriftMonitor
            driftmonitor.put(output_values)
            # return original response
            return ret

        return wrapper

    return monitor
