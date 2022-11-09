import numpy as np
import pandas as pd
import unittest

from metrics import FifoOverwriteDataFrame, convert_time_to_seconds


class TestFifoOverwriteDataFrame(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(
            FifoOverwriteDataFrame({"x": int}), FifoOverwriteDataFrame
        )

    def test_put(self):
        fifof = FifoOverwriteDataFrame({"x": int})
        fifof.put(np.arange(10))
        self.assertEqual(fifof.df.shape[0], 10)
        self.assertEqual(fifof.df.shape[1], 1)
        # more data
        fifof = FifoOverwriteDataFrame({"x": float, "y": float})
        fifof.put(np.random.rand(100, 2))
        self.assertEqual(fifof.df.shape[0], 100)
        self.assertEqual(fifof.df.shape[1], 2)

    def test_put_overwrite(self):
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=1)
        fifof.put(np.arange(2))
        self.assertEqual(fifof.df.iloc[0, 0], 1)
        #
        fifof = FifoOverwriteDataFrame({"x": int})
        fifof.put(np.arange(1001))
        self.assertEqual(fifof.df.iloc[0, 0], 1)
        self.assertEqual(fifof.df.iloc[-1, 0], 1000)
        #
        fifof = FifoOverwriteDataFrame({"x": str}, maxsize=3)
        fifof.put(["a", "b", "c", "d"])
        self.assertEqual(fifof.df.iloc[0, 0], "b")
        self.assertEqual(fifof.df.iloc[-1, 0], "d")

    def test_flush(self):
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=1)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        #
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=1)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 1)
        #
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=10)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 10)

    def test_flush_clear(self):
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=1, clear_at_flush=True)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=10, clear_at_flush=True)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0, 0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=1, clear_at_flush=True)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({"x": int}, maxsize=2, clear_at_flush=True)
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


from metrics import SummaryStatisticsMetrics, is_numeric_dtype


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
