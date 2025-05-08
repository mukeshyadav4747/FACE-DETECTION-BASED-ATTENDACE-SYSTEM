import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

# Timestamp setup
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")

# Auto refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# FizzBuzz display
st.title("FizzBuzz Auto Refresh Example")
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Attendance file path
csv_path = f"Attendance/Attendance_{date}.csv"

# Check and load attendance CSV
st.subheader("Today's Attendance")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.warning(f"No attendance file found for {date}")
