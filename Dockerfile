# Use the official Node 20 image (required for latest n8n)
FROM node:20-bookworm-slim

# Install Python and its tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install n8n globally via npm
RUN npm install -g n8n

# Install Python ML packages in smaller steps so Docker Desktop doesn't crash unpacking a massive layer!
RUN pip3 install --no-cache-dir --break-system-packages pandas numpy
RUN pip3 install --no-cache-dir --break-system-packages scikit-learn imbalanced-learn
RUN pip3 install --no-cache-dir --break-system-packages xgboost lightgbm
RUN pip3 install --no-cache-dir --break-system-packages mlflow
# Bypass strict config permissions (required for Windows Docker mounts)
ENV N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=false

# Switch to node user
USER node

# Start n8n correctly
CMD ["n8n", "start"]
