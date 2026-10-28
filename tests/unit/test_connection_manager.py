import threading
import time
from multiprocessing import get_context
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from databricks.sql.exc import Error
from dbt.adapters.contracts.connection import ConnectionState, Identifier, LazyHandle
from dbt_common.exceptions import DbtDatabaseError

from dbt.adapters.databricks.connections import (
    DatabricksConnectionManager,
    DatabricksDBTConnection,
    QueryContextWrapper,
)
from dbt.adapters.databricks.credentials import DatabricksCredentials
from dbt.adapters.databricks.dbr_capabilities import DBRCapabilities, DBRCapability
from dbt.adapters.databricks.utils import QueryTagsUtils, is_cluster_http_path


class TestDatabricksConnectionManager:
    def test_is_cluster_with_warehouse_path_no_cluster_id(self):
        """Test is_cluster() returns False for warehouse path with no cluster_id"""
        # Create a minimal connection manager with mock config
        mock_config = Mock()
        connection_manager = DatabricksConnectionManager(mock_config, get_context("spawn"))

        # Mock the connection
        mock_connection = Mock(spec=DatabricksDBTConnection)
        mock_connection.credentials = Mock(spec=DatabricksCredentials)
        mock_connection.credentials.cluster_id = None
        mock_connection.http_path = "sql/1.0/warehouses/abc123def456"

        with patch.object(
            connection_manager, "get_thread_connection", return_value=mock_connection
        ):
            assert connection_manager.is_cluster() is False

    def test_is_cluster_with_cluster_id_overrides_path(self):
        """Test is_cluster() returns False even when cluster_id is provided"""
        # Create a minimal connection manager with mock config
        mock_config = Mock()
        connection_manager = DatabricksConnectionManager(mock_config, get_context("spawn"))

        # Mock the connection with cluster_id set (overriding warehouse path)
        mock_connection = Mock(spec=DatabricksDBTConnection)
        mock_connection.credentials = Mock(spec=DatabricksCredentials)
        mock_connection.credentials.cluster_id = "cluster-123"
        mock_connection.http_path = "sql/1.0/warehouses/abc123def456"

        with patch.object(
            connection_manager, "get_thread_connection", return_value=mock_connection
        ):
            assert connection_manager.is_cluster() is False

    def test_is_cluster_http_path_function_warehouse_path(self):
        assert is_cluster_http_path("sql/1.0/warehouses/abc123def456", None) is False

    def test_is_cluster_http_path_function_cluster_path(self):
        assert is_cluster_http_path("sql/protocolv1/o/1234567890123456/", None) is True

    def test_is_cluster_http_path_function_cluster_id_overrides(self):
        assert is_cluster_http_path("sql/1.0/warehouses/abc123def456", "cluster-123") is False

    @patch("dbt.adapters.databricks.connections.DatabricksHandle.from_connection_args")
    @patch("dbt.adapters.databricks.connections.SqlUtils.prepare_connection_arguments")
    def test_open_calls_is_cluster_http_path_for_warehouse(
        self, mock_prepare_args, mock_from_connection_args
    ):
        """
        Test that open() method calls is_cluster_http_path with correct arguments for warehouse
        """
        # Create a minimal connection manager with mock config
        mock_config = Mock()
        connection_manager = DatabricksConnectionManager(mock_config, get_context("spawn"))

        # Mock the connection with proper timeout values
        mock_connection = Mock(spec=DatabricksDBTConnection)
        mock_connection.credentials = Mock(spec=DatabricksCredentials)
        mock_connection.credentials.cluster_id = None
        mock_connection.credentials.connect_retries = 1
        mock_connection.credentials.connect_timeout = 10
        mock_connection.credentials.query_tags = None
        mock_connection.credentials.compute = None  # default for real DatabricksCredentials
        mock_connection.credentials.host = "example.cloud.databricks.com"
        mock_connection.http_path = "sql/protocolv1/o/abc123def456"
        mock_connection.credentials.authenticate.return_value = Mock()
        mock_connection._query_header_context = None

        # Mock the handle creation
        mock_handle = Mock()
        mock_handle.session_id = "test_session"
        mock_from_connection_args.return_value = mock_handle

        mock_prepare_args.return_value = {}

        # Call open method
        connection_manager.open(mock_connection)

        # Verify that from_connection_args was called with is_cluster=False (warehouse path)
        mock_from_connection_args.assert_called_once()
        args, kwargs = mock_from_connection_args.call_args
        # Second argument (is_cluster) should be True for warehouse path with cluster_id
        assert args[1] is True


class TestCacheDbr:
    """Unit tests for _cache_dbr_capabilities."""

    HTTP_PATH = "sql/protocolv1/o/1234567890123456/cluster-abc"

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        DatabricksConnectionManager._dbr_capabilities_cache = {}
        yield
        DatabricksConnectionManager._dbr_capabilities_cache = {}

    @patch.object(DatabricksConnectionManager, "_query_dbr_version", return_value=None)
    def test_does_not_write_to_cache_when_version_is_none(self, mock_query):
        """Regression for #1398: a None version query result must not poison the cache."""
        creds = Mock(spec=DatabricksCredentials)
        creds.cluster_id = None

        DatabricksConnectionManager._cache_dbr_capabilities(creds, self.HTTP_PATH)

        assert self.HTTP_PATH not in DatabricksConnectionManager._dbr_capabilities_cache
        mock_query.assert_called_once_with(creds, self.HTTP_PATH)

    @patch.object(DatabricksConnectionManager, "_query_dbr_version", return_value=(15, 4))
    def test_writes_to_cache_when_version_is_known(self, mock_query):
        """When the version query succeeds, capabilities are cached correctly."""
        creds = Mock(spec=DatabricksCredentials)
        creds.cluster_id = None

        DatabricksConnectionManager._cache_dbr_capabilities(creds, self.HTTP_PATH)

        mock_query.assert_called_once_with(creds, self.HTTP_PATH)
        caps = DatabricksConnectionManager._dbr_capabilities_cache.get(self.HTTP_PATH)
        assert caps is not None
        assert caps.dbr_version == (15, 4)
        assert not caps.is_sql_warehouse
        assert caps.has_capability(DBRCapability.ICEBERG)

    @patch.object(DatabricksConnectionManager, "_query_dbr_version", return_value=(15, 4))
    def test_skips_write_when_already_cached(self, mock_query):
        """If the path is already in cache, the version query is never made."""
        creds = Mock(spec=DatabricksCredentials)
        creds.cluster_id = None
        existing = DBRCapabilities(dbr_version=(14, 3), is_sql_warehouse=False)
        DatabricksConnectionManager._dbr_capabilities_cache[self.HTTP_PATH] = existing

        DatabricksConnectionManager._cache_dbr_capabilities(creds, self.HTTP_PATH)

        mock_query.assert_not_called()
        assert DatabricksConnectionManager._dbr_capabilities_cache[self.HTTP_PATH] is existing

    def test_retry_succeeds_after_transient_failure(self):
        """Regression for #1398: after a transient None result, the next call must re-query."""
        creds = Mock(spec=DatabricksCredentials)
        creds.cluster_id = None

        with patch.object(DatabricksConnectionManager, "_query_dbr_version", return_value=None):
            DatabricksConnectionManager._cache_dbr_capabilities(creds, self.HTTP_PATH)

        with patch.object(DatabricksConnectionManager, "_query_dbr_version", return_value=(16, 2)):
            DatabricksConnectionManager._cache_dbr_capabilities(creds, self.HTTP_PATH)

        caps = DatabricksConnectionManager._dbr_capabilities_cache.get(self.HTTP_PATH)
        assert caps is not None
        assert caps.dbr_version == (16, 2)
        assert caps.has_capability(DBRCapability.ICEBERG)


class TestOpenSpogIntegration:
    """Verify DatabricksConnectionManager.open() invokes check_spog_preconditions."""

    def test_passes_all_http_paths_to_check(self):
        captured = {}

        def fake_check(*, host, http_paths):
            captured["host"] = host
            captured["http_paths"] = list(http_paths)
            return None

        with (
            patch(
                "dbt.adapters.databricks.connections.check_spog_preconditions",
                side_effect=fake_check,
            ),
            patch("dbt.adapters.databricks.connections.DatabricksHandle.from_connection_args"),
            patch(
                "dbt.adapters.databricks.connections.SqlUtils.prepare_connection_arguments",
                return_value={},
            ),
            patch(
                "dbt.adapters.databricks.connections.QueryConfigUtils.get_merged_query_tags",
                return_value={},
            ),
        ):
            fake_conn = Mock()
            fake_conn.state = "init"
            fake_conn.credentials = Mock(spec=DatabricksCredentials)
            fake_conn.credentials.host = "spog.example.com"
            fake_conn.credentials.http_path = "/sql/1.0/warehouses/default?o=64"
            fake_conn.credentials.compute = {"extra": {"http_path": "/sql/1.0/warehouses/alt?o=64"}}
            fake_conn.credentials.connect_timeout = None
            fake_conn.credentials.connect_retries = 1
            fake_conn.credentials.retry_all = False
            fake_conn.credentials.cluster_id = None
            fake_conn.credentials.authenticate = Mock(return_value=Mock())
            fake_conn.http_path = "/sql/1.0/warehouses/default?o=64"
            fake_conn._query_header_context = None

            DatabricksConnectionManager.open(fake_conn)

        assert captured["host"] == "spog.example.com"
        assert "/sql/1.0/warehouses/default?o=64" in captured["http_paths"]
        assert "/sql/1.0/warehouses/alt?o=64" in captured["http_paths"]


class TestRelease:
    def test_release_does_not_hold_lock_while_closing(self):
        profile = Mock()
        profile.credentials.reuse_connections = False
        connection_manager = DatabricksConnectionManager(profile, get_context("spawn"))
        conn = Mock(spec=DatabricksDBTConnection)
        connection_manager.set_thread_connection(conn)
        other_thread_got_lock = []

        def close_while_other_thread_takes_lock(_):
            def take_lock():
                acquired = connection_manager.lock.acquire(timeout=5)
                if acquired:
                    connection_manager.lock.release()
                other_thread_got_lock.append(acquired)

            other = threading.Thread(target=take_lock)
            other.start()
            other.join()

        with patch.object(
            DatabricksConnectionManager,
            "close",
            side_effect=close_while_other_thread_takes_lock,
        ) as mock_close:
            connection_manager.release()

        mock_close.assert_called_once_with(conn)
        assert other_thread_got_lock == [True]
        assert connection_manager.get_if_exists() is None

    def test_release_without_connection_is_noop(self):
        connection_manager = DatabricksConnectionManager(Mock(), get_context("spawn"))

        with patch.object(DatabricksConnectionManager, "close") as mock_close:
            connection_manager.release()

        mock_close.assert_not_called()


class TestReuseConnections:
    DEFAULT_PATH = "sql/1.0/warehouses/default"
    ALT_PATH = "sql/1.0/warehouses/alt"

    def manager(self, reuse_connections=True, connect_max_idle=None):
        profile = Mock()
        profile.credentials = Mock(spec=DatabricksCredentials)
        profile.credentials.reuse_connections = reuse_connections
        profile.credentials.connect_max_idle = connect_max_idle
        profile.credentials.http_path = self.DEFAULT_PATH
        profile.credentials.compute = {"alt": {"http_path": self.ALT_PATH}}
        profile.credentials.query_tags = None
        return DatabricksConnectionManager(profile, get_context("spawn"))

    def open_connection(self, connection_manager, last_used_at):
        conn = DatabricksDBTConnection(
            type=Identifier("databricks"),
            name="model.a",
            state=ConnectionState.OPEN,
            transaction_open=False,
            handle=Mock(),
            credentials=connection_manager.profile.credentials,
        )
        conn.http_path = self.DEFAULT_PATH
        conn.last_used_at = last_used_at
        connection_manager.set_thread_connection(conn)
        return conn

    def node(self, name, compute=None):
        config = Mock()
        config.get.return_value = compute
        config.extra = {}
        config.materialized = "table"
        return SimpleNamespace(name=name, relation_name=name, config=config)

    @pytest.fixture
    def mock_close(self):
        def close(conn):
            conn.state = ConnectionState.CLOSED

        with patch.object(DatabricksConnectionManager, "close", side_effect=close) as mock:
            yield mock

    def test_release_keeps_session(self, mock_close):
        connection_manager = self.manager()
        conn = self.open_connection(connection_manager, time.time())

        connection_manager.release()

        mock_close.assert_not_called()
        assert connection_manager.get_if_exists() is conn

    def test_next_node_reuses_session(self, mock_close):
        connection_manager = self.manager()
        conn = self.open_connection(connection_manager, time.time())

        result = connection_manager.set_connection_name("model.b", self.node("b"))

        mock_close.assert_not_called()
        assert result is conn
        assert conn.state == ConnectionState.OPEN
        assert conn._query_header_context.model_name == "b"

    def test_idle_session_is_recycled(self, mock_close):
        connection_manager = self.manager(connect_max_idle=30)
        conn = self.open_connection(connection_manager, time.time() - 31)

        connection_manager.set_connection_name("model.b", self.node("b"))

        mock_close.assert_called_once_with(conn)
        assert isinstance(conn._handle, LazyHandle)

    def test_session_is_recycled_after_failed_statement(self, mock_close):
        connection_manager = self.manager()
        conn = self.open_connection(connection_manager, time.time())

        with pytest.raises(DbtDatabaseError):
            with connection_manager.exception_handler("select 1"):
                raise Error("session expired")
        connection_manager.set_connection_name("model.b", self.node("b"))

        mock_close.assert_called_once_with(conn)

    def test_session_is_recycled_on_compute_change(self, mock_close):
        connection_manager = self.manager()
        conn = self.open_connection(connection_manager, time.time())

        connection_manager.set_connection_name("model.b", self.node("b", compute="alt"))

        mock_close.assert_called_once_with(conn)
        assert conn.http_path == self.ALT_PATH

    def test_idle_session_is_not_recycled_without_reuse(self, mock_close):
        connection_manager = self.manager(reuse_connections=False)
        self.open_connection(connection_manager, 0.0)

        connection_manager.set_connection_name("model.b", self.node("b"))

        mock_close.assert_not_called()

    @pytest.mark.parametrize("reuse_connections", [True, False])
    def test_add_query_tags_statements_only_with_reuse(self, reuse_connections):
        connection_manager = self.manager(reuse_connections=reuse_connections)
        conn = self.open_connection(connection_manager, time.time())
        connection_manager.set_connection_name("model.b", self.node("b"))

        with patch.object(
            DatabricksConnectionManager, "get_response", return_value=Mock(query_id="q")
        ):
            connection_manager.add_query("select 1")

        query_tags = conn.handle.execute.call_args.args[2]
        if reuse_connections:
            assert query_tags[QueryTagsUtils.DBT_MODEL_NAME_QUERY_TAG_KEY] == "b"
        else:
            assert query_tags is None

    def test_open_leaves_model_tags_off_the_session(self):
        connection_manager = self.manager()
        creds = connection_manager.profile.credentials
        creds.host = "example.cloud.databricks.com"
        creds.connect_timeout = None
        creds.connect_retries = 1
        creds.retry_all = False
        creds.cluster_id = None
        creds.authenticate = Mock(return_value=Mock())
        conn = self.open_connection(connection_manager, 0.0)
        conn.state = ConnectionState.INIT
        conn._query_header_context = QueryContextWrapper.from_context(self.node("a"))

        with (
            patch("dbt.adapters.databricks.connections.check_spog_preconditions"),
            patch("dbt.adapters.databricks.connections.DatabricksHandle.from_connection_args"),
            patch(
                "dbt.adapters.databricks.connections.SqlUtils.prepare_connection_arguments",
                return_value={},
            ) as mock_prepare,
        ):
            DatabricksConnectionManager.open(conn)

        session_tags = mock_prepare.call_args.args[3]
        assert QueryTagsUtils.DBT_MODEL_NAME_QUERY_TAG_KEY not in session_tags
        assert QueryTagsUtils.DBT_MATERIALIZED_QUERY_TAG_KEY not in session_tags

    def test_cleanup_all_closes_every_thread(self):
        connection_manager = self.manager()
        conns = [Mock(spec=DatabricksDBTConnection), Mock(spec=DatabricksDBTConnection)]
        conns[0].name, conns[1].name = "model.a", "model.b"
        connection_manager.thread_connections.update({(0, 1): conns[0], (0, 2): conns[1]})

        with patch.object(DatabricksConnectionManager, "close") as mock_close:
            connection_manager.cleanup_all()

        assert [c.args[0] for c in mock_close.call_args_list] == conns
        assert connection_manager.thread_connections == {}
