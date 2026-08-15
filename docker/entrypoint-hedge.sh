#!/bin/sh
set -eu

# Clean-mainline container entrypoint. No release-specific bootstrap or mutation.
if [ "$#" -eq 0 ]; then
    set -- freqtrade --version
fi
exec "$@"
