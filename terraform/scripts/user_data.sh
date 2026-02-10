#!/bin/bash
# =============================================================================
# user_data.sh — Bootstrap script for the GPU training node
#
# Executed once at first boot via EC2 user_data.
# Installs Lustre client drivers and mounts the FSx file system.
# =============================================================================

set -euxo pipefail

LOG_FILE="/var/log/user_data_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo " Starting bootstrap — $(date -u)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. System update
# ---------------------------------------------------------------------------
apt-get update -y

# ---------------------------------------------------------------------------
# 2. Install Lustre client drivers
#    Required to mount FSx for Lustre on Ubuntu 22.04
# ---------------------------------------------------------------------------
echo ">>> Installing Lustre client drivers..."
apt-get install -y lustre-client-modules-aws lustre-client-modules-$(uname -r) || true
apt-get install -y lustre-utils

# Verify Lustre kernel module is loadable
modprobe lustre || echo "WARNING: lustre module not loaded — may need reboot"

# ---------------------------------------------------------------------------
# 3. Create mount directory
# ---------------------------------------------------------------------------
MOUNT_POINT="${mount_point}"
mkdir -p "$MOUNT_POINT"
chown ubuntu:ubuntu "$MOUNT_POINT"

# ---------------------------------------------------------------------------
# 4. Mount FSx for Lustre
# ---------------------------------------------------------------------------
FSX_DNS="${fsx_dns_name}"
FSX_MOUNT_NAME="${fsx_mount_name}"

echo ">>> Mounting FSx: $FSX_DNS@tcp:/$FSX_MOUNT_NAME -> $MOUNT_POINT"
mount -t lustre -o noatime,flock "$FSX_DNS@tcp:/$FSX_MOUNT_NAME" "$MOUNT_POINT"

# Persist mount across reboots via /etc/fstab
FSTAB_ENTRY="$FSX_DNS@tcp:/$FSX_MOUNT_NAME $MOUNT_POINT lustre defaults,noatime,flock,_netdev 0 0"
if ! grep -q "$FSX_DNS" /etc/fstab; then
  echo "$FSTAB_ENTRY" >> /etc/fstab
  echo ">>> Added FSx mount to /etc/fstab"
fi

# ---------------------------------------------------------------------------
# 5. Verify mount
# ---------------------------------------------------------------------------
if mountpoint -q "$MOUNT_POINT"; then
  echo ">>> SUCCESS: FSx mounted at $MOUNT_POINT"
  df -h "$MOUNT_POINT"
else
  echo ">>> ERROR: FSx mount failed — check DNS and SG rules"
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Install Python dependencies for training script
# ---------------------------------------------------------------------------
echo ">>> Installing Python packages..."
sudo -u ubuntu pip install --no-cache-dir \
  mlflow \
  timm \
  pytorch-grad-cam \
  torchmetrics \
  albumentations

# ---------------------------------------------------------------------------
# 7. Set environment variables for convenience
# ---------------------------------------------------------------------------
ENV_FILE="/home/ubuntu/.bashrc"
{
  echo ""
  echo "# --- DL Training Environment ---"
  echo "export DATA_DIR=$MOUNT_POINT"
  echo "export NCCL_DEBUG=INFO"
  echo "export NCCL_SOCKET_IFNAME=ens"
  echo "export NCCL_P2P_LEVEL=NVL"
} >> "$ENV_FILE"
chown ubuntu:ubuntu "$ENV_FILE"

echo "=========================================="
echo " Bootstrap complete — $(date -u)"
echo "=========================================="
