from .models import Document, Principal


def authorized(doc: Document, principal: Principal) -> bool:
    return doc.tenant_id == principal.tenant_id and (
        not doc.allowed_roles or bool(doc.allowed_roles & principal.roles)
    )


def prefilter(documents: list[Document], principal: Principal) -> list[Document]:
    """Filter before any retrieval index or encoder sees the documents."""
    return [d for d in documents if authorized(d, principal)]
