from app.auth import create_access_token, hash_password
from app.models import User, WorkspaceInvitation
from sqlalchemy.dialects import postgresql


def test_workspace_inviter_foreign_key_uses_postgresql_uuid():
    dialect = postgresql.dialect()
    inviter_type = WorkspaceInvitation.__table__.c.inviter_id.type
    user_type = User.__table__.c.id.type

    assert inviter_type.compile(dialect=dialect) == "UUID"
    assert user_type.compile(dialect=dialect) == "UUID"


def test_workspace_invite_requires_admin(client, db_session, user):
    token = create_access_token(user.id)
    response = client.post(
        "/api/v1/workspaces/invite",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "invitee@utp.ac.pa", "workspace_name": "Engineering"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["message"] == "Admin access required"


def test_workspace_invite_creates_invitation_and_sends_email(client, db_session, monkeypatch):
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("password123"),
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    sent = {}

    def fake_send_email(to, subject, body, html=None):
        sent["to"] = to
        sent["subject"] = subject
        sent["body"] = body

    monkeypatch.setattr("app.routes.workspaces.send_email", fake_send_email)

    token = create_access_token(admin.id)
    response = client.post(
        "/api/v1/workspaces/invite",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "invitee@utp.ac.pa", "workspace_name": "Engineering"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "invitee@utp.ac.pa"
    assert payload["workspace_name"] == "Engineering"
    assert "invite_link" in payload
    assert payload["invite_link"].startswith("http")
    assert "token=" in payload["invite_link"]
    assert sent["to"] == "invitee@utp.ac.pa"
    assert "Invitation to join workspace" in sent["subject"]

    invitation = db_session.query(WorkspaceInvitation).filter_by(email="invitee@utp.ac.pa").first()
    assert invitation is not None
    assert invitation.workspace_name == "Engineering"
