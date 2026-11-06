# Security

EvidenceForge is a single-process service for local evaluation. Keep its port on loopback. Environment keys provide the demo identities. A shared deployment needs an identity provider, transport security, request limits and audit logs.

The server checks tenant equality and role intersection before building an index. The view cache is keyed by tenant and roles. Import replaces the corpus and clears the cache. A request already using a view can finish on that snapshot.

Set the import root to a directory controlled by the operator. The server restricts imports to JSONL files under that root, limits file and row size, checks each document and commits the replacement only after validation. A restart reloads the startup corpus.

Configure Ollama only at an endpoint that may receive the source excerpts. The draft filter checks source IDs. Treat document text and drafts as untrusted content and inspect claims against the excerpts.

Report security issues privately to the maintainer. Include the version, the request shape and the observed behavior. Redact keys and document contents.
