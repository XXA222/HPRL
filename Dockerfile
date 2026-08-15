# Freqtrade-Hedge clean-mainline reproducible Python 3.12 runtime.
ARG PYTHON_BASE_IMAGE=python:3.12.13-slim-bookworm
FROM ${PYTHON_BASE_IMAGE} AS base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    NO_COLOR=1 \
    FT_APP_ENV=docker \
    VIRTUAL_ENV=/opt/hedge-venv \
    PATH=/opt/hedge-venv/bin:$PATH

RUN python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates curl git libffi-dev libgomp1 libssl-dev sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/hedge-venv \
    && /opt/hedge-venv/bin/python -m pip install --upgrade pip wheel setuptools

WORKDIR /opt/freqtrade-hedge

COPY pyproject.toml requirements*.txt MANIFEST.in README.md LICENSE ./
COPY ft_client ./ft_client
COPY freqtrade ./freqtrade
COPY config_examples ./config_examples
COPY scripts ./scripts
COPY tools ./tools
COPY docker/entrypoint-hedge.sh /usr/local/bin/entrypoint-hedge

ARG INSTALL_DEVELOP=false
RUN chmod 0755 /usr/local/bin/entrypoint-hedge \
    && /opt/hedge-venv/bin/python -m pip install --no-cache-dir ./ft_client \
    && if [ "$INSTALL_DEVELOP" = "true" ]; then \
         /opt/hedge-venv/bin/python -m pip install --no-cache-dir '.[develop]'; \
       else \
         /opt/hedge-venv/bin/python -m pip install --no-cache-dir .; \
       fi \
    && /opt/hedge-venv/bin/python -m pip check \
    && /opt/hedge-venv/bin/python - <<'PY_GATE'
import pathlib
import sys
import freqtrade
assert sys.version_info[:2] == (3, 12), sys.version
root = pathlib.Path('/opt/freqtrade-hedge').resolve()
module = pathlib.Path(freqtrade.__file__).resolve()
assert root in module.parents, (root, module)
print('CLEAN_MAINLINE_IMPORT_GATE: PASS')
PY_GATE

RUN groupadd --gid 1000 ftuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash ftuser \
    && mkdir -p /opt/freqtrade-hedge/user_data \
    && chown -R ftuser:ftuser /opt/freqtrade-hedge /opt/hedge-venv

USER ftuser
EXPOSE 8080
VOLUME ["/opt/freqtrade-hedge/user_data"]
ENTRYPOINT ["/usr/local/bin/entrypoint-hedge"]
CMD ["freqtrade", "--version"]
