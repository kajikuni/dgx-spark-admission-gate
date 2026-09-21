# Security

Bind the gate to a trusted interface or put authentication and TLS in front of
it. The proxy deliberately does not add authentication. It forwards request
headers except hop-by-hop headers, `Host`, and `Content-Length`.

Please report vulnerabilities privately through GitHub's security advisory
feature. Do not include prompts, credentials, IP addresses, or production logs
in a public issue.
