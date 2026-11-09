"""A failed index replacement cannot leak a pending deletion into the next write."""

import sqlite3

import pytest

from deerflow.agents.memory.backends.deermem.deermem.core.retrieval import FTS5Retrieval


@pytest.mark.parametrize("failure", ["insert", "serialization"])
def test_failed_upsert_preserves_previous_row_and_next_transaction(tmp_path, failure):
    index = FTS5Retrieval(tmp_path / "facts.sqlite3")
    try:
        index.index_fact("existing", "original deployment preference")
        if failure == "insert":

            def deny_insert(action, table, _column, _database, _trigger):
                return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == "memory_fts" else sqlite3.SQLITE_OK

            index._conn.set_authorizer(deny_insert)
            expected_error = sqlite3.DatabaseError
            metadata = None
        else:
            metadata = {}
            metadata["cycle"] = metadata
            expected_error = ValueError

        with pytest.raises(expected_error):
            index.index_fact("existing", "replacement preference", fact_data=metadata)
        index._conn.set_authorizer(None)
        assert not index._conn.in_transaction
        index.index_fact("unrelated", "another independent memory")
        assert [fact["id"] for fact in index.search("original")] == ["existing"]
        assert not index.search("replacement")
        assert [fact["id"] for fact in index.search("independent")] == ["unrelated"]
    finally:
        index.close()
