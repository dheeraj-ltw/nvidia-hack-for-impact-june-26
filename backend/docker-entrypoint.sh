#!/bin/sh
# Fold any operator-provided root CAs into the system trust store before starting.
#
# Networks that intercept TLS (e.g. a Zscaler corporate proxy) re-sign HTTPS with their own
# root CA. Drop that root (PEM, any extension) into ./certs on the host — it's mounted here —
# and it becomes trusted system-wide via the standard ca-certificates mechanism. Python then
# uses the OS trust store (SSL_CERT_FILE, set in the image), so all outbound HTTPS verifies
# normally with no per-call configuration.
set -e

CERT_SRC=/usr/local/share/ca-certificates/extra
if [ -d "$CERT_SRC" ]; then
    # update-ca-certificates only picks up *.crt; normalize each PEM dropped in. Skip files
    # that aren't certificates (e.g. the README) so they don't pollute the trust store.
    found=0
    for cert in "$CERT_SRC"/*; do
        [ -f "$cert" ] || continue
        grep -ql "BEGIN CERTIFICATE" "$cert" || continue
        cp "$cert" "/usr/local/share/ca-certificates/extra-$(basename "$cert" | sed 's/\.[^.]*$//').crt"
        found=1
    done
    [ "$found" -eq 1 ] && update-ca-certificates >/dev/null 2>&1 || true
fi

exec "$@"
