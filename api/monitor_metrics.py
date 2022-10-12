from __future__ import annotations
from typing import Iterable, Union
from unicodedata import category
from xml.etree.ElementInclude import include
from prometheus_client import Summary, Counter, Gauge, Enum
import datetime
from pandas import Timestamp, Timedelta, Period, DateTime
from numpy import datetime64, timedelta64

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
        self.df = pd.concat(
            (self.df.iloc[1:] if y_size == self.maxsizes else self.df,
                pd.DataFrame(rows, columns=self.columns)),
            ignore_index=True)

        return self

    def flush(self)->pd.DataFrame:
        """
        If queue df is full, return copy.
        If self.clear_at_flush, clear df before returning copy.
        Else, return false.
        """
        if self.full(): return False
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

def date_objects_to_seconds(feature, dtype):
    """
    parse time objects to integer accuracy of seconds
    """
    if dtype == datetime:
        return feature.timestamp()
    elif dtype in [datetime64, timedelta64]:
        return (feature-timedelta64(datetime.datetime.min.timestamp())).total_seconds()
    elif dtype in [Timestamp, Timedelta, Period]:
        return date_objects_to_seconds(feature.to_numpy(), datetime64)
    else:
        raise ValueError(f'unsupported dtype for time: {dtype}')

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
        self.metric_name_prefix = metric_name_prefix
        
        # initialize metric names by calling summary statistics on an empty dataframe
        self.rownames = summary_statistics_function(pd.DataFrame(columns=dict)).index.values
        self.colnames =list(self.columns.keys())

        self.metrics = {} # store metric handles in a dict
        for colname in self.colnames:
            dtype = columns['colname']['dtype']
            for rowname in self.rownames:
                metric_key = '_'.join([colname, rowname])
                metric_name = metrics_name_prefix + key.replace(colname, '')
                metric_name = metric_name.replace('%', 'quantile')
                metric_description = ''
                if dtype in [datetime, Timestamp, datetime64]:
                    # e.g. date_of_birth, avg -> 'date_of_birth_timestamp_seconds_avg'
                    metric_name =  metric_name.replace('_timestamp','').replace('_seconds') + \
                        '_timestamp_seconds_' + colname
                elif dtype in [Timedelta, Period, timedelta64]:
                    # e.g. time_on_site, avg -> 'time_on_site_seconds_avg'
                    metric_name =  metric_name.replace('_seconds') + \
                        '_seconds_' + colname
                elif dtype == str:
                    # e.g. description, avg -> description_count_avg
                    metric_name =  metric_name + '_info_' + colname
                # TODO: how to check if value is of any possible integer or float type?
                elif dtype in [int, pd.Categorical, object]: 
                    # e.g. sex, avg -> sex_count_avg
                    metric_name + '_count_' + colname
                else: # numerical datatypes
                    metric_name = metric_name + colname

                self.metrics[metric_key] = Gauge(metric_name, metric_description)

    def set(self, df: pd.DataFrame):
        """
        Turn table of summary statistics to Prometheus metrics
        """
        sumstat = sumstat_function(latest_records)

        for colname in self.colnames:
            for rowname in self.rownames:
                metrics_handle = '_'.join([colname, rowname])
                metric_name = metrics_name_prefix + key.replace(metrics_name_prefix, '')
                metric_description = ''
                dtype = column['dtype']
                if dtype in [datetime, Timestamp, Timedelta, Period, datetime64, timedelta64]:
                    metric_value = date_objects_to_seconds(feature, dtype)
                else:
                    # TODO: define integerification to other data types
                    pass # add other options
                # omit nan values
                if metric_value is not None and not np.isnan(metric_value):
                    self.metrics[metrics_handle].set(metric_value)

        

    

# processing:
#   - time (total / hist )
#   - general resource usage
#   - request counter
# output:
#   - raw (if not text of some other weird datatype)
#   - if category
#   - hist/sumstat (a bit more private)
    - live_scoring