# Security policy

## Supported code

Security fixes are considered for the current default branch and the most
recent published release. Older snapshots may not receive fixes.

## Reporting a vulnerability

Use GitHub private vulnerability reporting or a private repository security
advisory. Do not open a public issue with exploit details, credentials, private
model endpoints, or unredacted recorded traffic.

Include:

- the affected version or revision;
- the operating system and Python version;
- a minimal reproduction;
- the expected and observed behavior;
- the likely impact; and
- any known workaround.

The proxy can handle API credentials and recorded client traffic. Remove real
secrets from reproductions. If a credential may have been exposed, rotate it
before sending the report.

No response-time or disclosure-time guarantee is made. Please allow time to
confirm the issue and prepare a fix before public disclosure.
