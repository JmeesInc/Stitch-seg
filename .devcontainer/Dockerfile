FROM nvcr.io/nvidia/tensorrt:26.01-py3

# 1. basic setup to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Node.js 20.x (Standard official method for Debian/Ubuntu)
# We also install 'sudo' and 'git' which are often missing but useful
# Also install libxcb1 for OpenCV compatibility
RUN apt-get update && \
    apt-get install -y curl git sudo libxcb1 libxcb-xinerama0 libxcb-render0 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# 4. Set the workspace
WORKDIR /workspaces