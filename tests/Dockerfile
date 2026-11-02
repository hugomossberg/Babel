# Build a disposable test image from the exact application image under test.
# Keep Node and repository test inputs out of the production runtime image.
ARG BABEL_TEST_BASE=babel:local
FROM ${BABEL_TEST_BASE}
RUN apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
