# Security

EvidenceForge is a local evaluation service with optional SQLite state for workers on one host. Keep its port on loopback. Environment keys provide the identities; an internet-facing deployment still needs an identity provider, TLS and proxy-level connection and abuse limits. Those external systems are not configured or claimed as validated here.

The server checks tenant equality and role intersection before building an index. The view cache is keyed by tenant and roles. Imports commit an entire validated corpus and a new revision transactionally. Workers refresh their views from the shared database before querying. A revision change detected before returning a response produces 409. Responses already sent to clients cannot be revoked, and this is not a guarantee of global instantaneous revocation.

Set the import root to a directory controlled by the operator. The server restricts imports to JSONL files under that root, limits file and row size, checks each document and commits the replacement only after validation. SQLite state survives a restart, including ACLs. Without a state file, the original process-local restart behavior remains. Keep the database on local disk, restrict its filesystem permissions and maintain tested backups. Database encryption is not provided by this application.

The model endpoint is restricted to loopback. Redirects and environment proxies are disabled. The model receives authorized excerpts only; the response must select an exact quotation from one of them. The application displays that source quotation, not unchecked model prose. It rejects mismatched IDs, invented quotations, malformed output and incomplete generations. This bounds what is displayed but does not prove relevance, truth of the original source, or general prompt-injection resistance.

The shared generation lease expires after a bounded interval so a crashed worker cannot block it indefinitely. Request quotas are fixed-window limits based on the host clock; deployments need synchronized clocks and external abuse protection. Audit metadata excludes source text and keys. Operator console access, database access and model-server administration are separate trust boundaries.

Report security issues privately to the maintainer. Include the version, the request shape and the observed behavior. Redact keys and document contents.
