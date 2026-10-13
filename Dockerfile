################
# BUILD STAGE
################

# Use a miniconda base image built on Alpine (linux distro)
FROM mambaorg/micromamba AS build

# Install conda-lock (to to make the venv)
# Install conda-pack (to make the venv relocatable)
RUN micromamba install -n base -c conda-forge conda-pack
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH="/opt/conda/bin:${PATH}"

# Copy the lock file into /tmp, and use it to install our venv.
# -n python-repo-template makes the env live in its own directory
COPY conda-lock.yml /tmp
RUN micromamba create -n python-repo-template -f /tmp/conda-lock.yml

# Set the current working directory to our /venv
# Bundle the already-made env into a single tarball that's relocated to /venv
# Reasons for this pack-unpack:
#   - Size, we get rid of cache/all-the-micromamba-info when we pack. The unpack makes sure it all works as expected.
WORKDIR /venv
RUN micromamba run -n base conda-pack -p /opt/conda/envs/python-repo-template -o /tmp/env.tar && \
    tar xf /tmp/env.tar && rm /tmp/env.tar
RUN /venv/bin/conda-unpack

################
# FINAL STAGE
################

# Use an ubuntu image which doesn't have micromamaba.
FROM python:3.12.13-slim

# Copy across our /venv folder.
COPY --from=build /venv /venv

# Prepend our venv path to PATH, such that our venv is searched first for executables e.g. python.
# In other words, /venv/bin/python is now the active Python.
ENV PATH="/venv/bin:${PATH}"

# Specify that Poetry shouldn't create a new .venv folder.
ENV POETRY_VIRTUALENVS_CREATE=false

# Subsequent RUN commands will now look in in /venv/bin
WORKDIR /opt/lab-in-the-loop

# Copy code etc. from the build machine (excluding .dockerignore dirs) to our image
COPY . .

# Install the package and it's dependencies into /venv.
# --without dev stops the install of poetry dev dependency group (see poetry.toml).
# Delete poetry's downloaded cache.
RUN poetry install --without dev && rm -rf /root/.cache/pypoetry

# Hugging Face Spaces runs the container as user 1000 and sends traffic to port 7860.
RUN useradd --uid 1000 --create-home user
USER 1000
EXPOSE 7860
ENV ALLOWED_ORIGINS=https://samhsoftware.github.io
CMD ["uvicorn", "lab_in_the_loop.api:app", "--host", "0.0.0.0", "--port", "7860"]
