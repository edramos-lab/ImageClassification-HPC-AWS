# =============================================================================
# iam.tf — IAM Role + Instance Profile for the GPU EC2 node
# =============================================================================

# -----------------------------------------------------------------------------
# IAM Role — trusted by EC2
# -----------------------------------------------------------------------------

resource "aws_iam_role" "gpu_node_role" {
  name = "${var.project_name}-gpu-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-gpu-node-role"
    Project = var.project_name
  }
}

# -----------------------------------------------------------------------------
# Policy Attachments — S3 and FSx full access
# -----------------------------------------------------------------------------

resource "aws_iam_role_policy_attachment" "s3_full_access" {
  role       = aws_iam_role.gpu_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "fsx_full_access" {
  role       = aws_iam_role.gpu_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonFSxFullAccess"
}

# -----------------------------------------------------------------------------
# Instance Profile — bridges the IAM role to the EC2 instance
# -----------------------------------------------------------------------------

resource "aws_iam_instance_profile" "gpu_node_profile" {
  name = "${var.project_name}-gpu-node-profile"
  role = aws_iam_role.gpu_node_role.name

  tags = {
    Name    = "${var.project_name}-gpu-node-profile"
    Project = var.project_name
  }
}
