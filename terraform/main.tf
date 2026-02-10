# =============================================================================
# main.tf — Provider, Data Sources, EC2 Compute (8x V100 GPU Node)
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used as prefix for resource naming"
  type        = string
  default     = "dl-hpc-training"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 Key Pair for SSH access"
  type        = string
}

# -----------------------------------------------------------------------------
# Data Source — Latest Deep Learning AMI (NVIDIA Driver, Ubuntu 22.04)
# -----------------------------------------------------------------------------

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# -----------------------------------------------------------------------------
# EC2 Instance — p3.16xlarge (8x NVIDIA V100, 64 vCPUs, 488GB RAM)
# -----------------------------------------------------------------------------

resource "aws_instance" "gpu_node" {
  ami                         = data.aws_ami.deep_learning.id
  instance_type               = "p3.16xlarge"
  key_name                    = var.key_pair_name
  iam_instance_profile        = aws_iam_instance_profile.gpu_node_profile.name
  placement_group             = aws_placement_group.cluster_pg.id
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.cluster_sg.id]
  associate_public_ip_address = true

  # Root volume: 500 GB gp3 with high IOPS for dataset I/O
  root_block_device {
    volume_size           = 500
    volume_type           = "gp3"
    iops                  = 6000
    throughput            = 400
    encrypted             = true
    delete_on_termination = true
  }

  # Bootstrap: install Lustre client and mount FSx
  user_data = base64encode(templatefile("${path.module}/scripts/user_data.sh", {
    fsx_dns_name   = aws_fsx_lustre_file_system.training_fs.dns_name
    fsx_mount_name = aws_fsx_lustre_file_system.training_fs.mount_name
    mount_point    = "/home/ubuntu/data"
  }))

  tags = {
    Name        = "${var.project_name}-gpu-node"
    Project     = var.project_name
    Environment = "training"
  }

  depends_on = [
    aws_fsx_lustre_file_system.training_fs
  ]
}
