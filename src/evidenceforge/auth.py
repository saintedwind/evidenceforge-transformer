import os

from .models import CorpusPolicy, Principal


def principals_from_env(raw: str | None = None) -> dict[str, Principal]:
    if raw is None:
        raw = os.getenv("EVIDENCEFORGE_API_KEYS", "")
    principals: dict[str, Principal] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        key, key_separator, identity = entry.partition("=")
        tenant, role_separator, roles = identity.partition(":")
        if not key_separator or not role_separator or not key or key in principals:
            raise ValueError("invalid or duplicate EVIDENCEFORGE_API_KEYS entry")
        if key != key.strip() or not key.isascii():
            raise ValueError("invalid API key format")
        policy = CorpusPolicy(tenant_id=tenant, allowed_roles=roles.split("|") if roles else [])
        principals[key] = Principal(policy.tenant_id, frozenset(policy.allowed_roles))
    return principals
