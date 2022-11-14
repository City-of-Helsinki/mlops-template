import numpy as np
import pandas as pd
import unittest

from metrics import (
    is_timedelta,
    is_timestamp,
    is_time,
    is_numeric,
    is_bool,
    is_str,
    is_categorical,
    is_object,
    get_dtypename,
    value_dtypename,
)

import datetime as dt

# data type samples
PY_NUMERIC = [[1, 1.0, 1.1], [2, 2.0, 2.2]]

NP_INT64 = np.ones(1).astype("int")
NP_FLOAT64 = NP_INT64.astype("float")
NP_BOOL = np.array([True]).astype("bool")
NP_STR = np.array(["test_string"]).astype("str")
NP_DT = np.array([np.datetime64("2020-10-01")])
NP_TD = np.array([np.timedelta64(1, "D")])

PD_INT64 = pd.DataFrame([1]).astype("int64")
PD_FLOAT64 = pd.DataFrame([1]).astype("float64")
PD_BOOL = pd.DataFrame([True]).astype("bool")
PD_BOOLEAN = pd.DataFrame([True]).astype("boolean")
PD_STR = pd.DataFrame(["string it is"]).astype("str")
PD_STRING = pd.DataFrame(["string it is"]).astype("string")
PD_DT = pd.to_datetime(["2020-10-1"])
PD_TD = pd.to_timedelta(["1d"])
PD_PERIOD = pd.DataFrame([pd.Period("2020Q1")])


class TestTypeChecks(unittest.TestCase):
    def test_dtypename(self):
        # built-in data types
        self.assertEqual(value_dtypename(1), "int")
        self.assertEqual(value_dtypename(1.0), "float")
        self.assertEqual(value_dtypename(True), "bool")
        self.assertEqual(value_dtypename("this is string"), "str")
        self.assertEqual(value_dtypename(dt.datetime.now()), "datetime")
        # numpy
        self.assertEqual(value_dtypename(NP_INT64[0]), "int64")
        self.assertEqual(value_dtypename(NP_FLOAT64[0]), "float64")
        self.assertEqual(value_dtypename(NP_BOOL[0]), "bool")
        self.assertEqual(value_dtypename(NP_STR[0]), "<U11")
        self.assertEqual(value_dtypename(NP_DT[0]), "datetime64[D]")
        self.assertEqual(value_dtypename(NP_TD[0]), "timedelta64[D]")
        # pandas
        self.assertEqual(value_dtypename(PD_INT64.iloc[0]), "int64")
        self.assertEqual(value_dtypename(PD_FLOAT64.iloc[0]), "float64")
        self.assertEqual(value_dtypename(PD_BOOL.iloc[0]), "bool")
        self.assertEqual(value_dtypename(PD_BOOLEAN.iloc[0]), "boolean")
        self.assertEqual(value_dtypename(PD_STR.iloc[0]), "object")
        self.assertEqual(value_dtypename(PD_STRING.iloc[0]), "string")
        self.assertEqual(value_dtypename(PD_DT[0]), "Timestamp")
        self.assertEqual(value_dtypename(PD_TD[0]), "Timedelta")
        self.assertEqual(value_dtypename(PD_PERIOD.iloc[0]), "period[Q-DEC]")

    def test_dtypename_checks(self):
        # built-in data types
        self.assertTrue(is_numeric(value_dtypename(1)))
        self.assertTrue(is_numeric(value_dtypename(1.0)))
        self.assertTrue(is_bool(value_dtypename(True)))
        self.assertTrue(is_str(value_dtypename("this is string")))
        self.assertTrue(is_time(value_dtypename(dt.datetime.now())))
        # numpy
        self.assertTrue(is_numeric(value_dtypename(NP_INT64[0])))
        self.assertTrue(is_numeric(value_dtypename(NP_FLOAT64[0])))
        self.assertTrue(is_bool(value_dtypename(NP_BOOL[0])))
        self.assertTrue(is_str(value_dtypename(NP_STR[0])))
        self.assertTrue(is_timestamp(value_dtypename(NP_DT[0])))
        self.assertTrue(is_timedelta(value_dtypename(NP_TD[0])))
        # pandas
        self.assertTrue(is_numeric(value_dtypename(PD_INT64.iloc[0])))
        self.assertTrue(is_numeric(value_dtypename(PD_FLOAT64.iloc[0])))
        self.assertTrue(is_bool(value_dtypename(PD_BOOL.iloc[0])))
        self.assertTrue(is_bool(value_dtypename(PD_BOOLEAN.iloc[0])))
        self.assertTrue(is_str(value_dtypename(PD_STR.iloc[0])))
        self.assertTrue(is_str(value_dtypename(PD_STRING.iloc[0])))
        self.assertTrue(is_timestamp(value_dtypename(PD_DT[0])))
        self.assertTrue(is_timedelta(value_dtypename(PD_TD[0])))
        self.assertTrue(is_timedelta(value_dtypename(PD_PERIOD.iloc[0])))
        # misc
        self.assertTrue(is_object(value_dtypename({"this": "is a dict"})))
        # negative
        self.assertFalse(is_time(value_dtypename(1)))
        self.assertFalse(is_timedelta(value_dtypename(PD_DT[0])))
        self.assertFalse(is_timestamp(value_dtypename(PD_TD[0])))
        self.assertFalse(is_bool(value_dtypename(1)))
        self.assertFalse(is_numeric(value_dtypename("string")))
        self.assertFalse(is_object(value_dtypename(1)))


from metrics import DriftQueue, convert_time_to_seconds


class TestDriftQueue(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(DriftQueue({"x": int}), DriftQueue)

    def test_put(self):
        fifof = DriftQueue({"x": int})
        fifof.put(np.arange(10))
        self.assertEqual(fifof.df.shape[0], 10)
        self.assertEqual(fifof.df.shape[1], 1)
        # more data
        fifof = DriftQueue({"x": float, "y": float})
        fifof.put(np.random.rand(100, 2))
        self.assertEqual(fifof.df.shape[0], 100)
        self.assertEqual(fifof.df.shape[1], 2)

    def test_put_overwrite(self):
        fifof = DriftQueue({"x": int}, maxsize=1)
        fifof.put(np.arange(2))
        self.assertEqual(fifof.df.iloc[0, 0], 1)
        #
        fifof = DriftQueue({"x": int})
        fifof.put(np.arange(1001))
        self.assertEqual(fifof.df.iloc[0, 0], 1)
        self.assertEqual(fifof.df.iloc[-1, 0], 1000)
        #
        fifof = DriftQueue({"x": str}, maxsize=3)
        fifof.put(["a", "b", "c", "d"])
        self.assertEqual(fifof.df.iloc[0, 0], "b")
        self.assertEqual(fifof.df.iloc[-1, 0], "d")

    def test_flush(self):
        fifof = DriftQueue({"x": int}, maxsize=1)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        #
        fifof = DriftQueue({"x": int}, maxsize=1)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 1)
        #
        fifof = DriftQueue({"x": int}, maxsize=10)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 10)

    def test_flush_clear(self):
        fifof = DriftQueue({"x": int}, maxsize=1, clear_at_flush=True)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = DriftQueue({"x": int}, maxsize=10, clear_at_flush=True)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = DriftQueue({"x": int}, maxsize=1, clear_at_flush=True)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = DriftQueue({"x": int}, maxsize=2, clear_at_flush=True)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret, False)
        self.assertEqual(fifof.df.shape[0], 1)


import time
import datetime as dt


class TestConvertTime(unittest.TestCase):
    def test_time(self):
        self.assertIsInstance(convert_time_to_seconds(time.time()), float)

    def test_datetime(self):
        self.assertIsInstance(convert_time_to_seconds(dt.date(1, 1, 1)), float)
        self.assertIsInstance(convert_time_to_seconds(dt.time(1, 1, 1)), float)
        self.assertIsInstance(convert_time_to_seconds(dt.datetime.min), float)
        self.assertIsInstance(
            convert_time_to_seconds(dt.datetime(2022, 1, 1, 12, 4, 4)), float
        )
        self.assertIsInstance(
            convert_time_to_seconds(dt.datetime.max - dt.datetime.min), float
        )
        self.assertEqual(convert_time_to_seconds(dt.timedelta(seconds=1.2)), 1.2)

    def test_numpy(self):
        self.assertIsInstance(
            convert_time_to_seconds(np.datetime64("2022-01-01")), float
        )
        self.assertIsInstance(
            convert_time_to_seconds(
                np.datetime64("2022-01-01") - np.datetime64("2021-01-01")
            ),
            float,
        )
        self.assertEqual(convert_time_to_seconds(np.timedelta64(1, "s")), 1.0)

    def test_pandas(self):
        self.assertIsInstance(
            convert_time_to_seconds(pd.to_datetime("2022-01-01")), float
        )
        self.assertIsInstance(
            convert_time_to_seconds(
                pd.to_datetime("2022-01-01") - pd.to_datetime("2021-01-01")
            ),
            float,
        )
        self.assertIsInstance(convert_time_to_seconds(pd.Period("4Q2005")), float)

    def test_float_integer(self):
        self.assertEqual(convert_time_to_seconds(1), 1.0)
        self.assertEqual(convert_time_to_seconds(1.2), 1.2)

    def test_string_parse(self):
        # timestamp
        self.assertIsInstance(convert_time_to_seconds("2022-01-01"), float)
        self.assertIsInstance(convert_time_to_seconds("2022/01/01"), float)
        self.assertIsInstance(convert_time_to_seconds("01-01-2022"), float)
        self.assertIsInstance(convert_time_to_seconds("01/01/2022"), float)
        self.assertIsInstance(
            convert_time_to_seconds("01/01/2022", pd_str_parse_format="%d/%m/%Y"), float
        )
        # timedelta
        self.assertIsInstance(convert_time_to_seconds("1 days 06:05:01.00003"), float)
        # period
        self.assertIsInstance(convert_time_to_seconds("4Q2005"), float)
        self.assertIsInstance(
            convert_time_to_seconds("01-01-2022", pd_str_parse_format="%d/%m/%Y"), float
        )

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            convert_time_to_seconds("this should raise an error!")
        with self.assertRaises(ValueError):
            convert_time_to_seconds(
                "01-01-2022",
                pd_str_parse_format="%d/%m/%Y",
                pd_infer_datetime_format=False,
            )
        with self.assertRaises(TypeError):
            convert_time_to_seconds(["this should raise an error"])
        with self.assertRaises(TypeError):
            convert_time_to_seconds({"as_should": "this"})

        self.assertIsInstance(
            convert_time_to_seconds({"as_should": "this"}, errors="ignore"), dict
        )
        self.assertTrue(
            np.isnan(convert_time_to_seconds({"as_should": "this"}, errors="coerce"))
        )


# TODO: test_convert_metric_name_to_promql

from metrics import convert_metric_name_to_promql


class TestConvertName(unittest.TestCase):
    def test_illegal_character_removal(self):
        self.assertEqual(convert_metric_name_to_promql("te$st", None), "te_st")
        self.assertEqual(convert_metric_name_to_promql("test%", None), "test")
        self.assertEqual(convert_metric_name_to_promql("&test", None), "test")

    def test_extra_undescore_removal(self):
        self.assertEqual(convert_metric_name_to_promql("te_st", None), "te_st")
        self.assertEqual(convert_metric_name_to_promql("te__st", None), "te_st")
        self.assertEqual(convert_metric_name_to_promql("test_", None), "test")
        self.assertEqual(convert_metric_name_to_promql("test__", None), "test")
        self.assertEqual(convert_metric_name_to_promql("_test", None), "test")
        self.assertEqual(convert_metric_name_to_promql("__test", None), "test")

    def test_prefix(self):
        # normal
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.date, prefix="test"),
            "test_page_visit_time_timestamp_seconds",
        )
        # prefixing underscore removal
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.date, prefix="_test"),
            "test_page_visit_time_timestamp_seconds",
        )
        # extra underscore removal
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.date, prefix="test_"),
            "test_page_visit_time_timestamp_seconds",
        )
        # illegal character removal
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.date, prefix="test#"),
            "test_page_visit_time_timestamp_seconds",
        )

    def test_suffix(self):
        # normal
        self.assertEqual(
            convert_metric_name_to_promql("value_generated", None, suffix="euros"),
            "value_generated_euros",
        )
        # illegal character removal
        self.assertEqual(
            convert_metric_name_to_promql("value_generated", None, suffix="euros_€"),
            "value_generated_euros",
        )

    def test_timestamp(self):
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.date),
            "page_visit_time_timestamp_seconds",
        )
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time_timestamp", dt.date),
            "page_visit_time_timestamp_seconds",
        )
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_timestamp_time", dt.date),
            "page_visit_time_timestamp_seconds",
        )
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time_seconds_timestamp", dt.date),
            "page_visit_time_timestamp_seconds",
        )

    def test_timedelta(self):
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time", dt.timedelta),
            "page_visit_time_seconds",
        )
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_time_seconds", dt.timedelta),
            "page_visit_time_seconds",
        )
        self.assertEqual(
            convert_metric_name_to_promql("page_visit_seconds_time", dt.timedelta),
            "page_visit_time_seconds",
        )

    def test_string(self):
        self.assertEqual(
            convert_metric_name_to_promql("optimization_function", str),
            "optimization_function_info",
        )

    def test_remove_metric_types(self):
        self.assertEqual(convert_metric_name_to_promql("test_map", None), "test")

    def test_remove_reserved_suffixes(self):
        convert_metric_name_to_promql("test_count", None)

        self.assertEqual(convert_metric_name_to_promql("test_count", None), "test")
        self.assertEqual(
            convert_metric_name_to_promql("test_count_count", None), "test"
        )
        # not removed from beginning or middle
        self.assertEqual(
            convert_metric_name_to_promql("count_test", None), "count_test"
        )
        self.assertEqual(
            convert_metric_name_to_promql("test_count_test", None), "test_count_test"
        )

    def test_counter_suffix(self):
        self.assertEqual(
            convert_metric_name_to_promql("test", None, is_counter=True), "test_total"
        )
        self.assertEqual(
            convert_metric_name_to_promql("test_total", None, is_counter=True),
            "test_total",
        )
        self.assertEqual(
            convert_metric_name_to_promql("total_test", None, is_counter=True),
            "total_test_total",
        )


from metrics import record_metrics_from_dict

from prometheus_client import Summary, Counter, Gauge, Enum, REGISTRY


def clean_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


class TestRecordDict(unittest.TestCase):
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
            "value": {
                "origin": "city-of-helsinki@github.com/ml-app",
                "branch": "main",
                "commit": "12345678",
            },
        },
        "model_update_time": {
            "value": dt.datetime.now(),
            "description": "model update workflow finish time",
            "type": "numeric",
        },
    }

    def test_metrics(self):
        clean_registry()
        m = record_metrics_from_dict(metrics=self.metrics)

        self.assertEqual(str(m[0]), "gauge:train_loss")

        self.assertEqual(str(m[2]), "stateset:optimizer")

        self.assertEqual(str(m[3]), "info:model_build_info")

        self.assertEqual(str(m[4]), "gauge:model_update_time_timestamp_seconds")

    def test_no_promql_convert(self):
        clean_registry()
        m = record_metrics_from_dict(
            metrics=self.metrics, convert_names_to_promql=False
        )

        self.assertEqual(str(m[4]), "gauge:model_update_time")


from metrics import SummaryStatisticsMetrics, is_numeric


class TestSummaryStatistics(unittest.TestCase):
    class stringable_object:
        def __str__():
            return "hello_world"

    class non_stringable_object:
        a = 1

    columns = {
        "numeric_float": float,
        "numeric_int": int,
        "bool": pd.BooleanDtype,
        "datetime": dt.datetime,
        "category": pd.CategoricalDtype,
        "string": str,
        "stringable_object": stringable_object,
        "non_stringable_object": non_stringable_object,
    }

    values1 = [
        [
            0.1,
            1,
            True,
            dt.datetime.now(),
            pd.Series(["a", "b", "c", "a"], dtype="category").iloc[0],
            "string",
            stringable_object(),
            non_stringable_object(),
        ]
    ]

    df1 = pd.DataFrame(columns=columns, data=values1)

    def test_init(self):
        clean_registry()
        ssm = SummaryStatisticsMetrics(columns=self.columns)
        for key in self.columns.keys():
            self.assertTrue(
                np.any(
                    [
                        metricname.startswith(convert_metric_name_to_promql(key))
                        for metricname in ssm.metrics.keys()
                    ]
                )
            )

    def test_set(self):
        clean_registry()
        ssm = SummaryStatisticsMetrics(columns=self.columns)

        ssm.set(self.df1)
        # reset
        ssm.set(self.df1)
