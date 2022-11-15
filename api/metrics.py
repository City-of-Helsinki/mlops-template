from __future__ import annotations
from typing import Iterable, Type, Union
from numbers import Number
from parser import ParserError
from prometheus_client import Summary, Counter, Gauge, Enum, Info
import pyarrow.feather as feather
import datetime as dt
import re

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


# / data types

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


# correct metric names to prometheus naming conventions:

# Prometheus naming conventions:
#
# suffix describing base unit, plural form
# _total for unitless count
# _unit_total for unit accumulating count
# _info for metadata
# _timestamp_seconds for timestamps
# rule of thumb: either sum() or avg() of metric must be meaningful, or metric has to be split up
# time -> seconds
# percent -> ratio  0-1
# bits, bytes -> bytes


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
    """

    ret = prefix + "_" + metric_name + "_" + suffix
    # must be lowercase
    ret = ret.lower().strip("_")

    # can't begin with non-alphabetical
    if not ret[:1].isalpha() and ret[:1].isnumeric():
        ret = "column" + ret

    if dtypename is None:
        dtypename = get_dtypename(dtype)

    if is_timestamp(dtypename):
        # e.g. date_of_birth -> 'date_of_birth_timestamp_seconds'
        ret = ret.replace("seconds", "").replace("timestamp", "") + "_timestamp_seconds"

    elif is_timedelta(dtypename):
        # e.g. time_on_site -> 'time_on_site_seconds'
        ret = ret.replace("seconds", "") + "_seconds"
    elif is_str(dtypename) and not category:
        # e.g. description -> description_count
        ret += "_info"
    else:  # add more exceptions if needed
        pass

    # remove metric types from metric name (a metric name should not contain these)
    for metric_type in ["gauge", "counter", "summary", "map"]:
        if not mask_type_aliases:  # remove
            ret = ret.replace(metric_type, "")
        else:  # mask
            ret = ret.replace(metric_type, metric_type + type_mask)

    ret = ret.strip("_")

    # remove reserved suffixes (a metric should not end with these)
    if not is_counter:
        for reserved_suffix in ["_count", "_sum", "_bucket", "_total"]:
            if not mask_reserved_suffixes:  # remove
                l = len(reserved_suffix)
                while ret.endswith(reserved_suffix):
                    ret = ret[:-l]
            elif ret.endswith(reserved_suffix):  # mask
                ret += suffix_mask
    elif not ret.endswith("_total"):  # however, counters should always end with _total
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
    ret = []  # return metric handles. In normal use these are not needed.

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
        if backup_file != "":
            try:  # initialize with stored data if available
                with open(backup_file, "rb") as f:
                    self.df = feather.read_feather(f)
            except FileNotFoundError:  # just initialize new
                self.df = pd.DataFrame(columns=columns)
        else:
            self.df = pd.DataFrame(columns=columns)

    def is_full(self) -> bool:
        if self.df.shape[0] >= self.maxsize:
            return True
        else:
            return False

    def put(self, rows: Union[np.ndarray, Iterable, dict, pd.DataFrame]) -> DriftQueue:
        """
        Put new items to queue. If full, overwrite the oldest value.
        Return reference to self.
        """
        y_size = self.df.shape[0]
        new_data = pd.DataFrame(rows, columns=self.columns)
        if new_data.shape[0] >= self.maxsize:
            new_data = new_data.iloc[-self.maxsize :]
        # TODO tsekkaa
        self.df = pd.concat(
            (self.df.iloc[1:] if y_size == self.maxsize else self.df, new_data),
            ignore_index=True,
        )

        # write backup for queue
        if self.backup_file != "":
            with open(self.backup_file, "wb") as f:
                feather.write_feather(self.df, f)
        return self

    def flush(self) -> pd.DataFrame:
        """
        Return queue contents as dataframe if full or non-full flush permitted,
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
        summary_statistics_function: function = lambda df: df.describe(
            include="all", datetime_is_numeric=True
        ),
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
        Create new metric if it does not already exist.
        """
        metric_key = f"{colname}_{rowname}"
        metric_description = f"calculated using summary statistics function {self.summary_statistics_function.__name__}"
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
        elif is_time(dtypename) or is_numeric(dtypename) or is_bool(dtypename):
            m = Gauge(metric_name, metric_description)
        else:  # string & rest
            m = Info(metric_name, metric_description)

        # record created metric in a dict
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
        self.categories_list = []
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
        # loop through dataframe and calculate summary statistics
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
                    try:  # other values should be convertable to string
                        metric_value = str(metric_value)
                        self.metrics[metric_key].info(metric_value)
                    except:  # try converting to float
                        try:
                            metric_value = float(metric_value)
                            self.metrics[metric_key].set(metric_value)
                        except:
                            pass  # do not record non-numeric, boolean, categorical or non-convertable to str
        return self
