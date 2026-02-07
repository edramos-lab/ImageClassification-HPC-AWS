# =============================================================================
# network.tf — VPC, Subnet, IGW, Routes, Placement Group, Security Group
# =============================================================================

# -----------------------------------------------------------------------------
# VPC
# -----------------------------------------------------------------------------

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# -----------------------------------------------------------------------------
# Public Subnet (single AZ — required for placement group locality)
# -----------------------------------------------------------------------------

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet"
  }
}

# -----------------------------------------------------------------------------
# Internet Gateway + Route Table
# -----------------------------------------------------------------------------

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public_rta" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public_rt.id
}

# -----------------------------------------------------------------------------
# Placement Group — cluster strategy for lowest latency (EFA + NCCL)
# -----------------------------------------------------------------------------

resource "aws_placement_group" "cluster_pg" {
  name     = "${var.project_name}-cluster-pg"
  strategy = "cluster"

  tags = {
    Name = "${var.project_name}-cluster-pg"
  }
}

# -----------------------------------------------------------------------------
# Security Group — SSH + unrestricted intra-cluster traffic (NCCL / EFA)
# -----------------------------------------------------------------------------

resource "aws_security_group" "cluster_sg" {
  name        = "${var.project_name}-cluster-sg"
  description = "Allow SSH and full intra-cluster traffic for NCCL/EFA"
  vpc_id      = aws_vpc.main.id

  # SSH from anywhere (restrict to your IP in production)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Intra-cluster: all TCP (NCCL control plane, data transfer)
  ingress {
    description = "All TCP - intra-cluster NCCL"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  # Intra-cluster: all UDP (NCCL data plane)
  ingress {
    description = "All UDP - intra-cluster NCCL"
    from_port   = 0
    to_port     = 65535
    protocol    = "udp"
    self        = true
  }

  # Intra-cluster: EFA traffic (custom protocol number 0 = all protocols)
  ingress {
    description = "EFA - all protocols intra-cluster"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # Egress: allow all outbound (package installs, S3, FSx, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-cluster-sg"
  }
}
