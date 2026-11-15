"""Small database layer for Eye on the Coffee Farm.

The same SQL runs on PostgreSQL (production, DATABASE_URL) and on SQLite (a laptop).
- SQL is written with "?" placeholders; for PostgreSQL they are changed to "%s".
- Booleans are stored as 0/1 integers and dates as UTC text "YYYY-MM-DD HH:MM:SS".
- One connection per web request. It is committed when the request ends well and
  rolled back when it fails.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import g, has_app_context

RAIZ = Path(__file__).resolve().parent
URL = os.environ.get("DATABASE_URL", "").strip()
if URL.startswith("postgres://"):
    URL = "postgresql://" + URL[len("postgres://"):]
IS_PG = URL.startswith("postgresql://")
SQLITE_PATH = os.environ.get("SQLITE_PATH", str(RAIZ / "sistema.db"))


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _connect():
    if IS_PG:
        try:
            import psycopg2 as driver          # psycopg2-binary
        except ImportError:
            import psycopg as driver           # psycopg 3
        return driver.connect(URL)
    conexion = sqlite3.connect(SQLITE_PATH, timeout=15)
    conexion.execute("PRAGMA foreign_keys = ON")
    return conexion


class Conn:
    def __init__(self):
        self.c = _connect()

    @staticmethod
    def _sql(sql):
        return sql.replace("?", "%s") if IS_PG else sql

    def all(self, sql, args=()):
        cur = self.c.cursor()
        cur.execute(self._sql(sql), tuple(args))
        columnas = [d[0] for d in cur.description]
        filas = [dict(zip(columnas, f)) for f in cur.fetchall()]
        cur.close()
        return filas

    def one(self, sql, args=()):
        filas = self.all(sql, args)
        return filas[0] if filas else None

    def run(self, sql, args=()):
        cur = self.c.cursor()
        cur.execute(self._sql(sql), tuple(args))
        n = cur.rowcount
        cur.close()
        return n

    def insert(self, sql, args=()):
        """INSERT and return the new id."""
        cur = self.c.cursor()
        if IS_PG:
            cur.execute(self._sql(sql) + " RETURNING id", tuple(args))
            nuevo = cur.fetchone()[0]
        else:
            cur.execute(sql, tuple(args))
            nuevo = cur.lastrowid
        cur.close()
        return nuevo

    def commit(self):
        self.c.commit()

    def rollback(self):
        self.c.rollback()

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


def get():
    """Connection of the current request (or a fresh one outside a request)."""
    if has_app_context():
        if "conn" not in g:
            g.conn = Conn()
        return g.conn
    return Conn()


def cerrar_request(error=None):
    conn = g.pop("conn", None)
    if conn is not None:
        if error is not None:
            conn.rollback()
        conn.close()


# ------------------------------------------------------------------ schema
PK = "SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"

TABLAS = [
    """CREATE TABLE IF NOT EXISTS settings (
         name TEXT PRIMARY KEY,
         value TEXT NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS users (
         id {PK},
         username TEXT NOT NULL,
         username_lc TEXT NOT NULL UNIQUE,
         full_name TEXT NOT NULL DEFAULT '',
         password_hash TEXT NOT NULL,
         role TEXT NOT NULL DEFAULT 'grower',
         active INTEGER NOT NULL DEFAULT 1,
         created_at TEXT NOT NULL,
         last_login TEXT)""",
    f"""CREATE TABLE IF NOT EXISTS farms (
         id {PK},
         name TEXT NOT NULL,
         municipality TEXT NOT NULL,
         department TEXT NOT NULL,
         owner_id INTEGER NOT NULL REFERENCES users(id),
         created_at TEXT NOT NULL,
         UNIQUE (owner_id, name))""",
    f"""CREATE TABLE IF NOT EXISTS plots (
         id {PK},
         farm_id INTEGER NOT NULL REFERENCES farms(id),
         name TEXT NOT NULL,
         variety TEXT NOT NULL DEFAULT 'unknown',
         area_ha DOUBLE PRECISION NOT NULL,
         planted_year INTEGER,
         notes TEXT NOT NULL DEFAULT '',
         created_at TEXT NOT NULL,
         UNIQUE (farm_id, name))""",
    f"""CREATE TABLE IF NOT EXISTS inspections (
         id {PK},
         plot_id INTEGER NOT NULL REFERENCES plots(id),
         created_by INTEGER NOT NULL REFERENCES users(id),
         created_at TEXT NOT NULL,
         sick INTEGER NOT NULL,
         probability_sick DOUBLE PRECISION NOT NULL,
         confidence_pct INTEGER NOT NULL,
         confidence_level TEXT NOT NULL,
         damaged_pct DOUBLE PRECISION,
         rust_level INTEGER NOT NULL DEFAULT 0,
         doubtful_cut INTEGER NOT NULL DEFAULT 0,
         model_version TEXT NOT NULL,
         response_ms INTEGER NOT NULL,
         image_b64 TEXT NOT NULL,
         features TEXT NOT NULL DEFAULT '[]',
         notes TEXT NOT NULL DEFAULT '',
         status TEXT NOT NULL DEFAULT 'pending_review',
         reviewed_by INTEGER REFERENCES users(id),
         reviewed_at TEXT,
         true_label TEXT,
         review_note TEXT NOT NULL DEFAULT '')""",
    f"""CREATE TABLE IF NOT EXISTS recommendations (
         id {PK},
         inspection_id INTEGER NOT NULL REFERENCES inspections(id),
         variety_type TEXT NOT NULL,
         action TEXT NOT NULL,
         text TEXT NOT NULL,
         status TEXT NOT NULL DEFAULT 'pending',
         note TEXT NOT NULL DEFAULT '',
         updated_by INTEGER REFERENCES users(id),
         created_at TEXT NOT NULL,
         updated_at TEXT NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS feedback (
         id {PK},
         user_id INTEGER NOT NULL REFERENCES users(id),
         inspection_id INTEGER REFERENCES inspections(id),
         kind TEXT NOT NULL DEFAULT 'feedback',
         answer_correct INTEGER,
         message TEXT NOT NULL DEFAULT '',
         status TEXT NOT NULL DEFAULT 'open',
         response TEXT NOT NULL DEFAULT '',
         created_at TEXT NOT NULL,
         resolved_by INTEGER REFERENCES users(id))""",
    f"""CREATE TABLE IF NOT EXISTS event_log (
         id {PK},
         ts TEXT NOT NULL,
         level TEXT NOT NULL DEFAULT 'info',
         event TEXT NOT NULL,
         user_id INTEGER,
         detail TEXT NOT NULL DEFAULT '',
         ms INTEGER)""",
    "CREATE INDEX IF NOT EXISTS ix_insp_plot ON inspections (plot_id)",
    "CREATE INDEX IF NOT EXISTS ix_insp_created ON inspections (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_log_ts ON event_log (ts)",
    "CREATE INDEX IF NOT EXISTS ix_reco_insp ON recommendations (inspection_id)",
]


def crear_tablas():
    conn = Conn()
    try:
        for sql in TABLAS:
            conn.run(sql)
        conn.commit()
    finally:
        conn.close()
