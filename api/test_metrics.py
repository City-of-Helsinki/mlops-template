import numpy as np
import pandas as pd
import unittest

from metrics import FifoOverwriteDataFrame, convert_time_to_seconds


class testFifoOverwriteDataFrame(unittest.TestCase):

    def test_init(self):
        self.assertIsInstance(FifoOverwriteDataFrame({'x':int}), FifoOverwriteDataFrame)
    def test_put(self):
        fifof = FifoOverwriteDataFrame({'x':int})
        fifof.put(np.arange(10))
        self.assertEqual(fifof.df.shape[0], 10)
        self.assertEqual(fifof.df.shape[1], 1)
        # more data
        fifof = FifoOverwriteDataFrame({'x':float, 'y':float})
        fifof.put(np.random.rand(100, 2))
        self.assertEqual(fifof.df.shape[0], 100)
        self.assertEqual(fifof.df.shape[1], 2)
    def test_put_overwrite(self):
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=1)
        fifof.put(np.arange(2))
        self.assertEqual(fifof.df.iloc[0,0], 1)
        #
        fifof = FifoOverwriteDataFrame({'x':int})
        fifof.put(np.arange(1001))
        self.assertEqual(fifof.df.iloc[0,0], 1)
        self.assertEqual(fifof.df.iloc[-1,0], 1000)
        # 
        fifof = FifoOverwriteDataFrame({'x':str}, maxsize=3)
        fifof.put(['a','b','c','d'])
        self.assertEqual(fifof.df.iloc[0,0], 'b')
        self.assertEqual(fifof.df.iloc[-1,0], 'd')
    def test_flush(self):
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=1)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        #
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=1)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0,0], 1)
        self.assertEqual(fifof.df.shape[0], 1)
        #
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=10)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0,0], 1)
        self.assertEqual(fifof.df.shape[0], 10)
    def test_flush_clear(self):
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=1, clear_at_flush=True)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0,0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=10, clear_at_flush=True)
        fifof.put(range(11))
        ret = fifof.flush()
        self.assertEqual(ret.iloc[0,0], 1)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=1, clear_at_flush=True)
        ret = fifof.flush()
        self.assertEqual(ret, False)
        self.assertEqual(fifof.df.shape[0], 0)
        #
        fifof = FifoOverwriteDataFrame({'x':int}, maxsize=2, clear_at_flush=True)
        fifof.put([1])
        ret = fifof.flush()
        self.assertEqual(ret, False)
        self.assertEqual(fifof.df.shape[0], 1)

import time
import datetime as dt

class test_convert_time_to_seconds(unittest.TestCase):
    
    def test_time(self):
        self.assertIsInstance(convert_time_to_seconds(time.time()), int)

    def test_datetime(self):
        self.assertIsInstance(convert_time_to_seconds(dt.date(1,1,1)), int)
        self.assertIsInstance(convert_time_to_seconds(dt.time(1,1,1)), int)
        self.assertIsInstance(convert_time_to_seconds(dt.datetime.min), int)
        self.assertIsInstance(convert_time_to_seconds(dt.datetime(2022, 1, 1, 12, 4, 4)), int)
        self.assertIsInstance(convert_time_to_seconds(dt.datetime.max-dt.datetime.min), int)
        self.assertEqual(convert_time_to_seconds(dt.timedelta(seconds = 1.2)), 1)

    def test_numpy(self):
        self.assertIsInstance(convert_time_to_seconds(np.datetime64('2022-01-01')), int)
        self.assertIsInstance(convert_time_to_seconds(np.datetime64('2022-01-01')-np.datetime64('2021-01-01')), int)
        self.assertEqual(convert_time_to_seconds(np.timedelta64(1,'s')),1)

    def test_pandas(self):
        self.assertIsInstance(convert_time_to_seconds(pd.to_datetime('2022-01-01')), int)
        self.assertIsInstance(convert_time_to_seconds(pd.to_datetime('2022-01-01')-pd.to_datetime('2021-01-01')), int)
        self.assertIsInstance(convert_time_to_seconds(pd.Period('4Q2005')), int)

    def test_float_integer(self):
        self.assertEqual(convert_time_to_seconds(1), 1)
        self.assertEqual(convert_time_to_seconds(1.2), 1)

    def test_invalid_formats(self):
        with self.assertRaises(ValueError):
            convert_time_to_seconds('this should raise an error!')