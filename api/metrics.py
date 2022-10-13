from __future__ import annotations
from typing import Iterable, Type, Union
from prometheus_client import Summary, Counter, Gauge, Enum
import datetime
import re

import numpy as np
import pandas as pd

# For some metrics, we may want to keep a FIFO que for
# calculating summary statistics over fixed number of records
# instead of a period of time that is the standard for Prometheus:

class FifoOverwriteDataFrame:
    """
    A FIFO Queue for storing maxsize latest items.
    
    Properties:
     - if full, will replace the oldest value with the newest
     - can only be emptied if completely full (flush)
     - optionally clear (drop all) at flush
    """
    def __init__(self, columns: dict, maxsize: int = 1000, clear_at_flush: bool = False):
        self.is_full = False
        self.maxsize = maxsize
        self.columns = columns
        self.clear_at_flush = clear_at_flush
        self.df = pd.DataFrame(columns=columns)

    def put(self, rows: Union[np.ndarray, Iterable, dict, pd.DataFrame])->FifoOverwriteDataFrame:
        """
        Put new items to queue. If full, overwrite the oldest value.
        Return reference to self.
        """
        y_size = self.df.shape[0]
        new_data = pd.DataFrame(rows, columns=self.columns)
        if new_data.shape[0] >= self.maxsize:
            new_data = new_data.iloc[-self.maxsize:]
        self.df = pd.concat(
            (self.df.iloc[1:] if y_size == self.maxsize else self.df,
                new_data),
            ignore_index=True)
        if self.df.shape[0] == self.maxsize:
            self.is_full = True
        else:
            self.is_full = False

        return self

    def flush(self)->pd.DataFrame:
        """
        If queue df is full, return copy.
        If self.clear_at_flush, clear df before returning copy. Queue is only cleared if full.
        Else, return false.
        """
        if not self.is_full: return False
        ret = self.df.copy()
        if self.clear_at_flush: self.df.drop(self.df.index, inplace=True)
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


def model_creation_metrics(metrics: dict) -> None:
    """
    Pass pre-recorded metrics from dict to Prometheus.
    This allows recording two types of metrics
        - numeric (int, float, etc.)
        - categorical
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
            'type': str -> 'numeric' or 'category',
            'categories': [str], e.g. ['A', 'B', 'C']. only required if 'type' = category
        }
    }

    Example: 

    metrics = {
        'train_loss':{'value':0.95, 'description': 'training loss (MSE)', 'type':'numeric'},
        'test_loss':{'value':0.96, 'description': 'test loss (MSE)', 'type':'numeric'},
        'optimizer':{'value':random.choice(['SGM', 'RMSProp', 'Adagrad']),
            'description':'ml model optimizer function',
            'type': 'category',
            'categories':['SGD', 'RMSProp', 'Adagrad', 'Adam']}
    """

    for metric_name in metrics.keys():
        metric = model_creation_metrics[metric_name]
        if metric['type'] == 'numeric':
            g = Gauge(metric_name, metric['description'])
            g.set(metric['value'])
        elif metric['type'] == 'category':
            s = Enum(
                metric_name,
                metric['description'],
                states = metric['categories']
            )
            s.state(metric['value'])

# TODO: PROMEHEUS:
# input:
#   - raw values (if not text or some other weird datatype)
#   - hist/sumstat (a bit more private)


def convert_time_to_seconds(feature, dtype):
    """
    parse time objects to integer accuracy of seconds
    """
    if dtype == datetime:
        return feature.timestamp()
    elif dtype in [datetime64, timedelta64]:
        return (feature-timedelta64(datetime.datetime.min.timestamp())).total_seconds()
    elif dtype in [Timestamp, Timedelta, Period]:
        return convert_time_to_seconds(feature.to_numpy(), datetime64)
    else:
        raise ValueError(f'unsupported dtype for time: {dtype}')

def create_promql_metric_name(metric_name: str,
                                dtype: Type,
                                prefix: str = '',
                                suffix: str = '',
                                is_counter = False):
    """
    Create a promql compatible name for a metric.
    Note that this does not perfectly ensure good naming,
    but may help when auto-generating metrics.

    See https://prometheus.io/docs/practices/naming/ for naming conventions.
    """

    ret = prefix.rstrip('_') # may not begin with underscore
    if isinstance(dtype(),(datetime, pd.Timestamp, np.datetime64)):
        # e.g. date_of_birth -> 'date_of_birth_timestamp_seconds'
        metric_name =  metric_name.strip('_seconds').strip('_timestamp').strip('_seconds') + \
            '_timestamp_seconds'
    elif isinstance(dtype(),(pd.Timedelta, pd.Period,np.timedelta64)):
        # e.g. time_on_site -> 'time_on_site_seconds'
        metric_name = metric_name.strip('_seconds') + '_seconds'
    elif isinstance(dtype(), (str, np.string_, np.unicode_, np.dtype('O'),
                        pd.object, pd.CategoricalDtype, pd.StringDtype)):
        # e.g. description -> description_count
        metric_name += '_info'
    else: # add more exceptions if needed
        pass
    
    ret += metric_name + '_' + suffix

    # remove metric types from metric name (a metric name should not contain these)
    for metric_type in ['gauge', 'counter', 'summary', 'map']:
        metric_name = metric_name.replace(metric_type, '')
    
    # remove reserved suffixes (a metric should not end with these)
    for metric_suffix in ['_count', '_sum', '_bucket', '_total']:
        metric_name = metric_name.strip(metric_suffix)

    # however, counters should always have _total suffix
    if is_counter: metric_name += '_total'

    # clean extra underscores
    metric_name = ''.join(['_' + val.strip('_') if val != '' else '' for val in metric_name.split('_')]).rstrip('_')
    
    # clean non-alphanumericals and underscores
    metric_name = re.sub('[a-zA-Z_:][a-zA-Z0-9_:]*', '_', s)

    return metric_name


class SummaryStatisticsMetrics:
    """
    Class for wrapping metrics based on dataframe summary statistics, 
    for example FifoOverwriteDataFrame.flush()
    """

    def init(self,
            columns: dict,
            summary_statistics_function: function = lambda df: df.describe(include = 'all'),
            metrics_name_prefix: str = ''):
        """
        metric_name_prefix is a common prefix for metric names, e.g. 'input_feature_'
        """
        self.columns = columns
        self.summary_statistics_function = summary_statistics_function
        
        # initialize metric names by calling summary statistics on an empty dataframe
        self.rownames = summary_statistics_function(pd.DataFrame(columns=dict)).index.values
        self.colnames =list(self.columns.keys())

        self.metrics = {} # store metric handles in a dict
        for colname in self.colnames:
            dtype = columns['colname']['dtype']
            for rowname in self.rownames:
                metric_key = '_'.join([colname, rowname])
                metric_description = f'calculated using summary statistics function {summary_statistics_function.__name__}'
                metric_name = create_promql_metric_name(metric_name=colname,
                    dtype=dtype,
                    prefix = metrics_name_prefix,
                    postfix = rowname
                )
                # if category, create Enum
                if isinstance(dtype(), (str, np.string_, np.unicode_, np.dtype('O'),
                        pd.object, pd.CategoricalDtype, pd.StringDtype)):
                    g = Enum(
                        metric_name,
                        metric_description,
                        states = []
                    )
                # else Gauge
                else:
                    g = Gauge(metric_name, metric['description'])
                self.metrics[metric_key] = g

    def set(self, df: pd.DataFrame):
        """
        Set metrics to given value
        """
        sumstat_df = self.summary_statistics_function(df)
        # loop through metrics and 
        for colname in self.colnames:
            for rowname in self.rownames:
                metric_key = '_'.join([colname, rowname])
                metric_value = sumstat_df.loc[rowname, colname]
                if isinstance(metric_value,(datetime, pd.Timestamp,
                        pd.Timedelta, pd.Period, np.datetime64, np.timedelta64)):
                    # convert all time formats to integer seconds
                    metric_value = convert_time_to_seconds(feature, dtype)
                elif isinstance(metric_value, (bool, np.bool_, pd.BooleanDtype)):
                    metric_value = 1 if metric_value else 0
                elif isinstance(metric_value, (str, np.string_, np.unicode_, np.dtype('O'),
                        pd.object, pd.CategoricalDtype, pd.StringDtype)):
                    metric_value = str(metric_value)
                else:
                    pass # add other options if needed
                # omit nan values
                if metric_value is not None and not np.isnan(metric_value):
                    self.metrics[metric_key].set(metric_value)

        

    

# processing:
#   - time (total / hist )
#   - general resource usage
#   - request counter
# output:
#   - raw (if not text of some other weird datatype)
#   - if category
#   - hist/sumstat (a bit more private)
#   - live_scoring



