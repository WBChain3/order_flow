#!/usr/bin/env python3
"""
Downloads free Databento historical sample data for CME futures.

Usage:
    python download_databento_sample.py

This script requires a free Databento account and API key.
Set the DATABENTO_API_KEY environment variable before running.

The downloaded parquet files will be placed in the data/ directory
(which is gitignored) for local development and testing.

Reference: https://databento.com/docs
"""

import os
import sys

DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY")

if not DATABENTO_API_KEY:
    print("Error: DATABENTO_API_KEY environment variable not set.")
    print()
    print("To get a free API key:")
    print("  1. Sign up at https://databento.com/")
    print("  2. Generate an API key from your dashboard")
    print("  3. Set the environment variable:")
    print("     export DATABENTO_API_KEY='your-key-here'")
    print()
    print("Then run this script again.")
    sys.exit(1)

print("Databento API key found.")
print()
print("To download sample data programmatically, use the databento Python SDK:")
print()
print("  pip install databento")
print()
print("  import databento as db")
print('  client = db.Historical(key=DATABENTO_API_KEY)')
print()
print("  # Download NQ futures tick data for a single day")
print('  data = client.timeseries.get_range(')
print('      dataset="GLBX.MDP3",')
print('      symbols=["NQ"],')
print('      schema="trades",')
print('      start="2025-01-02",')
print('      end="2025-01-03",')
print("  )")
print()
print("  # Convert to our standard tick schema")
print('  data.to_parquet("data/nq_20250102.parquet")')
print()
print("For development without real data, use the synthetic tick generator:")
print("  from footprint.io._synthetic import SyntheticTickGenerator")
print("  ticks = generator.generate_candle()")
