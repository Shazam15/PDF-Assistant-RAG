import networkx as nx

from app.auth import create_access_token, hash_password
from app.metrics import record_query_response_time
from app.models import Document, User
from app.rag import graph_builder


def test_admin_stats_requires_admin(client, auth_headers):
    response = client.get("/api/v1/admin/stats", headers=auth_headers)

    assert response.status_code == 403
    assert response.json()["error"]["message"] == "Admin access required"


def test_admin_stats_returns_aggregate_metrics(client, db_session):
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    regular = User(
        username="regular",
        email="regular@example.com",
        hashed_password=hash_password("password123"),
    )
    db_session.add_all([admin, regular])
    db_session.commit()
    db_session.refresh(admin)
    db_session.refresh(regular)

    db_session.add_all(
        [
            Document(
                user_id=regular.id,
                filename="first.pdf",
                original_name="first.pdf",
                file_size=100,
                status="ready",
            ),
            Document(
                user_id=regular.id,
                filename="notes.txt",
                original_name="notes.txt",
                file_size=50,
                status="ready",
            ),
        ]
    )
    db_session.commit()

    record_query_response_time(0.25)

    token = create_access_token(admin.id)
    response = client.get(
        "/api/v1/admin/stats",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_users"] == 2
    assert payload["total_pdfs_uploaded"] == 1
    assert payload["average_query_response_time_ms"] > 0
    assert payload["query_count"] >= 1
    assert payload["disk_space_usage"]["total_bytes"] > 0
    assert payload["disk_space_usage"]["usage_percent"] >= 0


def test_graph_endpoints_require_admin(client, auth_headers, ready_document):
    list_response = client.get("/api/v1/admin/graphs", headers=auth_headers)
    detail_response = client.get(
        f"/api/v1/admin/graphs/{ready_document.id}",
        headers=auth_headers,
    )

    assert list_response.status_code == 403
    assert detail_response.status_code == 403


def test_admin_can_list_and_view_another_users_graph(
    client,
    db_session,
    user,
    ready_document,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(graph_builder.settings, "GRAPH_PERSIST_DIR", str(tmp_path))
    admin = User(
        username="graph-admin",
        email="graph-admin@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    graph = nx.Graph()
    graph.add_node(
        "ORG:openai",
        name="OpenAI",
        label="ORG",
        mentions=3,
        pages=[1, 2],
        chunks=[0, 1],
    )
    graph.add_node(
        "PRODUCT:atlas",
        name="ATLAS",
        label="PRODUCT",
        mentions=2,
        pages=[2],
        chunks=[1],
    )
    graph.add_edge(
        "ORG:openai",
        "PRODUCT:atlas",
        weight=2,
        pages=[2],
        chunks=[1],
    )
    graph_builder.save_graph(graph, str(user.id), str(ready_document.id))

    headers = {"Authorization": f"Bearer {create_access_token(admin.id)}"}
    inventory = client.get("/api/v1/admin/graphs", headers=headers)
    detail = client.get(
        f"/api/v1/admin/graphs/{ready_document.id}?max_nodes=10",
        headers=headers,
    )

    assert inventory.status_code == 200
    assert inventory.json()["total"] == 1
    assert inventory.json()["items"][0]["owner_username"] == user.username
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["document_name"] == ready_document.original_name
    assert payload["returned_node_count"] == 2
    assert payload["returned_edge_count"] == 1
    assert payload["nodes"][0]["degree"] == 1
    assert payload["edges"][0]["weight"] == 2
    assert "chunks" not in payload["nodes"][0]
    assert "chunks" not in payload["edges"][0]
    assert "path" not in payload
