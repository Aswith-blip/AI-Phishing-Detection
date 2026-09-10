import sqlite3
from datetime import datetime


DATABASE = "phishing_scans.db"


def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            phishing_probability REAL NOT NULL,
            legitimate_probability REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_scan(
    url,
    prediction,
    phishing_probability,
    legitimate_probability,
    risk_score,
    risk_level
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO scans (
            url,
            prediction,
            phishing_probability,
            legitimate_probability,
            risk_score,
            risk_level,
            scanned_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        url,
        prediction,
        phishing_probability,
        legitimate_probability,
        risk_score,
        risk_level,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    conn.commit()
    conn.close()


def get_recent_scans(limit=20):

    conn = get_connection()

    scans = conn.execute("""
        SELECT *
        FROM scans
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    conn.close()

    return scans


def get_statistics():

    conn = get_connection()

    total = conn.execute("""
        SELECT COUNT(*)
        FROM scans
    """).fetchone()[0]

    phishing = conn.execute("""
        SELECT COUNT(*)
        FROM scans
        WHERE prediction = 'PHISHING WEBSITE'
    """).fetchone()[0]

    legitimate = conn.execute("""
        SELECT COUNT(*)
        FROM scans
        WHERE prediction = 'LEGITIMATE WEBSITE'
    """).fetchone()[0]

    high_risk = conn.execute("""
        SELECT COUNT(*)
        FROM scans
        WHERE risk_level = 'HIGH RISK'
    """).fetchone()[0]

    medium_risk = conn.execute("""
        SELECT COUNT(*)
        FROM scans
        WHERE risk_level = 'MEDIUM RISK'
    """).fetchone()[0]

    low_risk = conn.execute("""
        SELECT COUNT(*)
        FROM scans
        WHERE risk_level = 'LOW RISK'
    """).fetchone()[0]

    if total > 0:
        phishing_percentage = round(
            (phishing / total) * 100,
            2
        )
    else:
        phishing_percentage = 0

    conn.close()

    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "phishing_percentage": phishing_percentage
    }