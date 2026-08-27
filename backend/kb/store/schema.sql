-- kb storage schema.
--
-- One SQLite file holds the whole knowledge base: documents, chunks, the FTS5
-- index that provides BM25, and the dense vectors. Keeping lexical and dense
-- side by side in one transactional store is what makes hybrid retrieval cheap
-- and consistent — there is no second system to fall out of sync.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    name            TEXT PRIMARY KEY,
    description     TEXT NOT NULL DEFAULT '',
    embedding_model TEXT,
    embedding_dim   INTEGER,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    collection     TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    title          TEXT NOT NULL,
    uri            TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    byte_size      INTEGER NOT NULL DEFAULT 0,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    n_chunks       INTEGER NOT NULL DEFAULT 0,
    language       TEXT,
    author         TEXT,
    published_at   TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}'
);

-- Re-ingesting the same bytes into the same collection is a no-op, not a duplicate.
CREATE UNIQUE INDEX IF NOT EXISTS documents_collection_hash
    ON documents (collection, content_hash);
CREATE INDEX IF NOT EXISTS documents_collection ON documents (collection);
CREATE INDEX IF NOT EXISTS documents_uri        ON documents (collection, uri);
CREATE INDEX IF NOT EXISTS documents_source     ON documents (collection, source_type);

CREATE TABLE IF NOT EXISTS chunks (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    document_id     TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    collection      TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'prose',
    locator         TEXT NOT NULL,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT NOT NULL,
    document_title  TEXT NOT NULL DEFAULT '',
    source_type     TEXT,
    heading_context TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS chunks_document   ON chunks (document_id, ordinal);
CREATE INDEX IF NOT EXISTS chunks_collection ON chunks (collection);
CREATE INDEX IF NOT EXISTS chunks_source     ON chunks (collection, source_type);

-- BM25 comes from FTS5 for free. `porter` stemming makes "retrieving" match
-- "retrieval"; the extra columns are indexed so a heading or document title can
-- contribute to the lexical score, weighted below the body text at query time.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    heading_context,
    document_title,
    content       = 'chunks',
    content_rowid = 'seq',
    tokenize      = "porter unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text, heading_context, document_title)
    VALUES (new.seq, new.text, new.heading_context, new.document_title);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, heading_context, document_title)
    VALUES ('delete', old.seq, old.text, old.heading_context, old.document_title);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, heading_context, document_title)
    VALUES ('delete', old.seq, old.text, old.heading_context, old.document_title);
    INSERT INTO chunks_fts (rowid, text, heading_context, document_title)
    VALUES (new.seq, new.text, new.heading_context, new.document_title);
END;

-- Vectors live as raw little-endian float32 blobs. `norm` is stored so cosine
-- similarity is a dot product with no per-query normalisation pass.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id   TEXT PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    norm       REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS embeddings_collection ON embeddings (collection, model);

-- Every retrieval is logged. This is what powers the "which parts of the corpus
-- actually get used" heatmap, and it turns production traffic into candidate
-- evaluation queries.
CREATE TABLE IF NOT EXISTS retrieval_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    query      TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    chunk_id   TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    score      REAL NOT NULL,
    strategy   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS retrieval_events_chunk ON retrieval_events (chunk_id);
CREATE INDEX IF NOT EXISTS retrieval_events_coll  ON retrieval_events (collection, created_at);
