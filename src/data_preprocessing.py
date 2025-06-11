# src/data_preprocessing.py
import pandas as pd
import numpy as np
import re

def extract_amount(x):
    match = re.search(r'\(\$ (-?[\d,]+\.\d+)\)', str(x))
    if match:
        return float(match.group(1).replace(',', ''))
    return np.nan

def normalize_str(s):
    return s.strip().lower() if isinstance(s, str) else s

def clean_contracts_df(df):
    df.dropna(inplace=True)
    df['ARTIST NET_CLEANED'] = df['ARTIST NET'].apply(extract_amount)
    df['PRESENTER_CLEAN'] = df['PRESENTER'].apply(normalize_str)
    df['VENUE_CLEAN'] = df['VENUE NAME'].apply(normalize_str)
    df['AGENT_CLEAN'] = df['AGENT'].apply(normalize_str)
    df['EVENT_DATE_CLEAN'] = pd.to_datetime(df['FIRST EVENT DATE'], errors='coerce')
    return df

def clean_bluecard_df(df):
    df.dropna(subset=['FIRST EVENT DATE'], inplace=True)
    df['CREATED DATE'] = pd.to_datetime(df['CREATED DATE'], errors='coerce')
    df.sort_values('CREATED DATE', inplace=True)
    df['Blue Card Rolling Count'] = df.reset_index().index + 1
    df['CREATED YEAR'] = df['CREATED DATE'].dt.year
    df['PRESENTER_CLEAN'] = df['PRESENTER'].apply(normalize_str)
    df['VENUE_CLEAN'] = df['VENUE NAME'].apply(normalize_str)
    df['AGENT_CLEAN'] = df['AGENT'].apply(normalize_str)
    df['EVENT_DATE_CLEAN'] = pd.to_datetime(df['FIRST EVENT DATE'], errors='coerce')
    return df

def clean_presenter_df(df):
    df.dropna(subset=['PHYSICAL CITY/STATE'], inplace=True)
    df['PHYSICAL CITY/STATE'].replace(' --', np.nan, inplace=True)
    return df

def clean_lead_df(df):
    df.dropna(subset=['AGENT', 'HOME OFFICE', 'EVENT DATE', 'CLOSED DATE', 'REFERRAL SOURCE'], inplace=True)
    df['EVENT DATE'] = pd.to_datetime(df['EVENT DATE'], errors='coerce')
    return df
 
