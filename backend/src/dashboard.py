import os
import sqlite3

from flask import Flask, redirect, render_template_string, url_for

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bol_khata.db")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>Bol-Khata Call Analytics</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f3f4f6; color: #1f2937; margin: 0; padding: 2rem; }
        h1 { text-align: center; margin-bottom: 2rem; color: #111827; font-weight: 800; }
        .dashboard { display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
        .card { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; flex: 1; min-width: 250px; max-width: 350px; }
        .card h2 { margin: 0 0 1rem 0; font-size: 1.25rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; }
        .card .value { font-size: 5rem; font-weight: 900; margin: 0; line-height: 1; }
        .value.total { color: #3b82f6; }
        .value.success { color: #10b981; }
        .value.failed { color: #ef4444; }
    </style>
</head>
<body>
    <h1>Bol-Khata Agent Analytics</h1>
    <div class="dashboard">
        <div class="card">
            <h2>Total Calls</h2>
            <p class="value total">{{ total }}</p>
        </div>
        <div class="card">
            <h2>Successful</h2>
            <p class="value success">{{ success }}</p>
        </div>
        <div class="card">
            <h2>Failed</h2>
            <p class="value failed">{{ failed }}</p>
        </div>
    </div>
    <form action="/reset" method="POST" style="text-align: center; margin-top: 3rem;">
        <button type="submit" style="background-color: #ef4444; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: bold; font-size: 1rem; cursor: pointer; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#dc2626'" onmouseout="this.style.backgroundColor='#ef4444'">Reset Dashboard</button>
    </form>
</body>
</html>
"""


def get_metrics():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM call_logs")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM call_logs WHERE outcome = 'Successful'")
        success = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM call_logs WHERE outcome = 'Failed'")
        failed = c.fetchone()[0]

        return total, success, failed
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0, 0, 0
    finally:
        if conn:
            conn.close()


@app.route("/")
def dashboard():
    total, success, failed = get_metrics()
    return render_template_string(
        HTML_TEMPLATE, total=total, success=success, failed=failed
    )


@app.route("/reset", methods=["POST"])
def reset():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM call_logs")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database reset error: {e}")
    finally:
        if conn:
            conn.close()
    return redirect(url_for("dashboard"))


def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS call_logs
               (id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
