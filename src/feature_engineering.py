import pandas as pd
import numpy as np

def add_match_keys(contracts_df, bluecard_df):
    """
    Adds normalized keys and generates MATCH_KEY for merging.
    """
    def normalize_str(s):
        return s.strip().lower() if isinstance(s, str) else s

    for df in [contracts_df, bluecard_df]:
        df["PRESENTER_CLEAN"] = df["PRESENTER"].apply(normalize_str)
        df["VENUE_CLEAN"] = df["VENUE NAME"].apply(normalize_str)
        df["AGENT_CLEAN"] = df["AGENT"].apply(normalize_str)
        df["EVENT_DATE_CLEAN"] = pd.to_datetime(df["FIRST EVENT DATE"], errors="coerce")
        df["MATCH_KEY"] = (
            df["PRESENTER_CLEAN"] + "_" +
            df["VENUE_CLEAN"].fillna("na") + "_" +
            df["EVENT_DATE_CLEAN"].dt.strftime('%Y-%m-%d')
        )

    return contracts_df, bluecard_df

def enrich_contracts(contracts_df, bluecard_df):
    """
    Enriches contracts with bluecard metadata and engineered features.
    """
    # Deduplicate bluecard by earliest CREATED DATE
    bluecard_df["CREATED DATE"] = pd.to_datetime(bluecard_df["CREATED DATE"], errors="coerce")
    bluecards_deduped = (
        bluecard_df.sort_values("CREATED DATE")
        .drop_duplicates("MATCH_KEY")
        .set_index("MATCH_KEY")
    )

    # Merge with contracts
    contracts_df = contracts_df.set_index("MATCH_KEY")
    enriched = contracts_df.join(
        bluecards_deduped[["# BLUE CARD", "CREATED DATE", "EVENT_DATE_CLEAN"]],
        how="left", rsuffix="_BLUECARD"
    ).reset_index()

    # Feature 1: Overdue flags
    enriched['OVERDUE_DEPOSIT_FLAG'] = enriched['OVERDUE DEPOSIT'].astype(str).str.lower().eq('yes').astype(int)
    enriched['OVERDUE_SIGNATURE_FLAG'] = enriched['OVERDUE SIGNATURE'].astype(str).str.lower().eq('yes').astype(int)

    # Feature 2: Status-based risk
    status_keywords = ['cancel', 'overdue', 'pending', 'hold']
    enriched['STATUS_RISK_FLAG'] = enriched['STATUS'].astype(str).str.lower().apply(
        lambda x: int(any(k in x for k in status_keywords))
    )

    # Feature 3: Financial Delta
    enriched["FINANCIAL_DELTA"] = (
        enriched["$GROSS"] -
        enriched["ARTIST NET_CLEANED"] -
        enriched["$ECE TOTAL COMMISSION"]
    )

    # Feature 4: Issue to Event Time
    enriched["ISSUE_TO_EVENT_DAYS"] = (
        pd.to_datetime(enriched["EVENT_DATE_CLEAN"], errors="coerce") -
        pd.to_datetime(enriched["ISSUE DATE"], errors="coerce")
    ).dt.days

    # Feature 5: First-Time Presenter
    presenter_counts = enriched["PRESENTER_CLEAN"].value_counts()
    enriched["IS_FIRST_TIME_PRESENTER"] = enriched["PRESENTER_CLEAN"].map(presenter_counts) == 1

    # Feature 6: Financial Anomaly Detection
    std_delta = enriched["FINANCIAL_DELTA"].std()
    enriched["FINANCIAL_ANOMALY_FLAG"] = (enriched["FINANCIAL_DELTA"].abs() > 3 * std_delta).astype(int)

    return enriched

def calculate_risk_score(df):
    """
    Combines engineered features into a final risk score.
    """
    df['RISK_SCORE'] = (
        (df['$DEPOSIT DUE AMOUNT'] < 500).astype(int) * 3 +
        df['IS_FIRST_TIME_PRESENTER'].astype(int) * 2 +
        (df['ISSUE_TO_EVENT_DAYS'] < 30).astype(int) * 4 +
        df['OVERDUE_DEPOSIT_FLAG'] * 2 +
        df['OVERDUE_SIGNATURE_FLAG'] * 1 +
        df['FINANCIAL_ANOMALY_FLAG'] * 3 +
        df['STATUS_RISK_FLAG'] * 2
    )
    return df
 
