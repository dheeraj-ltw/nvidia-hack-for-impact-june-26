# Trusted root CAs

Drop a root CA certificate (PEM format) into this folder to have the backend container
trust it on startup. Use this when you're behind a TLS-intercepting proxy (e.g. a Zscaler
corporate gateway) that re-signs HTTPS with its own root — without it, outbound calls from
the container (webhooks, ElevenLabs, NVIDIA) fail with `CERTIFICATE_VERIFY_FAILED`.

The folder is mounted read-only at `/usr/local/share/ca-certificates/extra`. On container
start, every cert here is folded into the system trust store via `update-ca-certificates`,
and Python uses that store (`SSL_CERT_FILE`), so verification stays fully enabled.

## macOS — export your proxy's root CA

```bash
security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain \
  > certs/corporate-root.crt
```

(Replace `Zscaler` with your proxy's CA name.) Then restart the backend:

```bash
docker compose up -d api
```

Certificate files here are git-ignored — only this README is tracked.
