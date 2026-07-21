"""
text_to_sql.py  —  Natural-language → validated read-only SQL
=============================================================
The JD names Snowflake + SQL, and the prep guide's hardest enterprise question
is: "how do you let an LLM query a financial data warehouse WITHOUT it running a
destructive or runaway query?" This module is the answer, demoable end to end.

THE SAFETY LAYERS (what the interview probes for):
  1. Read-only by construction   — only SELECT is ever executed.
  2. A SQL guard/linter          — blocks DROP/DELETE/UPDATE/INSERT/ALTER/etc.,
                                    multiple statements, and comment-injection.
  3. Curated schema, not raw DDL — the LLM sees a small semantic map of allowed
                                    tables/columns, not the whole database.
  4. Row-limit enforcement       — a LIMIT is injected to prevent scanning the
                                    whole warehouse (cost + performance guard).
  5. Deterministic execution     — the DB does the math; the LLM only writes SQL.

Runs on a tiny local SQLite "finance warehouse" so it's fully demoable offline.
Swap the executor for a Snowflake read-only service account in production — the
guard logic is identical. That's the line to say in the interview.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .providers import get_llm


# ============================================================================
# 1. Curated schema (the LLM sees THIS, not raw DDL)
# ============================================================================
SCHEMA_MAP = """
Table: balance_sheet
  - line_item   TEXT    -- e.g. 'Cash & Bank', 'Loans Outstanding', 'Total Assets'
  - category    TEXT    -- one of: 'asset', 'liability', 'equity'
  - amount      INTEGER -- value in USD
  - fiscal_year INTEGER -- e.g. 1995

Only SELECT queries are allowed. Never modify data.
"""


# ============================================================================
# 2. The SQL guard (blocks anything that isn't a safe single SELECT)
# ============================================================================
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|merge|grant|"
    r"revoke|attach|pragma|vacuum|reindex)\b", re.IGNORECASE)


@dataclass
class SQLValidation:
    ok: bool
    reason: str
    safe_sql: str = ""


def validate_sql(sql: str, max_rows: int = 100) -> SQLValidation:
    """Return a safe, executable SELECT or reject with a reason."""
    s = sql.strip().rstrip(";").strip()

    # strip code fences the LLM might add
    s = re.sub(r"^```sql\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    if not s:
        return SQLValidation(False, "empty query")

    # single statement only (block stacked queries)
    if ";" in s:
        return SQLValidation(False, "multiple statements are not allowed")

    # must start with SELECT (or WITH ... SELECT)
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return SQLValidation(False, "only SELECT queries are allowed")

    # no data-modifying / DDL keywords anywhere
    if _FORBIDDEN.search(s):
        bad = _FORBIDDEN.search(s).group(0)
        return SQLValidation(False, f"forbidden keyword: {bad.upper()}")

    # block SQL comments (injection vector)
    if "--" in s or "/*" in s:
        return SQLValidation(False, "comments are not allowed")

    # enforce a row limit (cost/performance guard)
    if not re.search(r"\blimit\b", s, re.IGNORECASE):
        s = f"{s} LIMIT {max_rows}"

    return SQLValidation(True, "ok", safe_sql=s)


# ============================================================================
# 3. NL → SQL via the LLM (constrained to the curated schema)
# ============================================================================
SQL_SYSTEM = """You translate a finance question into a single SQLite SELECT query.

RULES:
- Output ONLY the SQL. No prose, no markdown, no explanation.
- Use ONLY the tables/columns in the provided schema.
- SELECT only. Never write INSERT/UPDATE/DELETE/DROP or any modifying statement.
- One statement. No semicolons beyond the end. No comments.
"""


def nl_to_sql(question: str) -> str:
    user = f"Schema:\n{SCHEMA_MAP}\n\nQuestion: {question}\n\nSQL:"
    raw = get_llm().generate(SQL_SYSTEM, user)
    return raw.strip()


# ============================================================================
# 4. Executor (SQLite demo warehouse; swap for Snowflake read-only in prod)
# ============================================================================
def build_demo_warehouse() -> sqlite3.Connection:
    """A tiny in-memory finance table mirroring the sample balance sheet."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE balance_sheet (
        line_item TEXT, category TEXT, amount INTEGER, fiscal_year INTEGER)""")
    rows = [
        ("Cash & Bank", "asset", 5000, 1995),
        ("Interest Bearing Deposits", "asset", 8000, 1995),
        ("Loans Outstanding (Gross)", "asset", 84000, 1995),
        ("Loan Loss Reserve", "asset", -7000, 1995),
        ("Net Loans Outstanding", "asset", 77000, 1995),
        ("Other Current Assets", "asset", 500, 1995),
        ("Total Assets", "asset", 90500, 1995),
        ("Short-term Borrowings", "liability", 30000, 1995),
        ("Total Liabilities", "liability", 45000, 1995),
        ("Total Equity", "equity", 45500, 1995),
    ]
    conn.executemany("INSERT INTO balance_sheet VALUES (?,?,?,?)", rows)
    conn.commit()
    return conn


@dataclass
class SQLResult:
    question: str
    generated_sql: str
    safe_sql: str
    ok: bool
    reason: str
    rows: list


def ask_sql(question: str, conn: sqlite3.Connection = None) -> SQLResult:
    """Full flow: NL -> SQL -> validate -> (read-only) execute."""
    conn = conn or build_demo_warehouse()
    generated = nl_to_sql(question)
    v = validate_sql(generated)
    if not v.ok:
        return SQLResult(question, generated, "", False, v.reason, [])
    try:
        cur = conn.execute(v.safe_sql)
        rows = cur.fetchall()
        return SQLResult(question, generated, v.safe_sql, True, "ok", rows)
    except Exception as e:
        return SQLResult(question, generated, v.safe_sql, False, f"execution error: {e}", [])


if __name__ == "__main__":
    conn = build_demo_warehouse()

    print("=== Legitimate questions ===")
    for q in ["What is the total assets amount?",
              "Show all asset line items and their amounts",
              "What is total equity?"]:
        r = ask_sql(q, conn)
        print(f"\nQ: {q}")
        print(f"   SQL: {r.safe_sql or r.generated_sql}")
        print(f"   ok={r.ok} rows={r.rows if r.ok else r.reason}")

    print("\n=== The guard blocking attacks (direct validate) ===")
    for bad in ["DROP TABLE balance_sheet",
                "SELECT * FROM balance_sheet; DELETE FROM balance_sheet",
                "UPDATE balance_sheet SET amount=0",
                "SELECT * FROM balance_sheet -- comment injection"]:
        v = validate_sql(bad)
        print(f"   {'BLOCKED' if not v.ok else 'ALLOWED'}: {bad[:45]:47} → {v.reason}")
