# =============================================================================
# storage.tf — S3 Bucket (dataset origin) + FSx for Lustre (high-perf mount)
# =============================================================================

# -----------------------------------------------------------------------------
# S3 Bucket — persistent storage for Kaggle datasets
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "dataset_bucket" {
  bucket_prefix = "${var.project_name}-data-"
  force_destroy = true      # Allow terraform destroy to remove non-empty bucket

  tags = {
    Name    = "${var.project_name}-dataset-bucket"
    Project = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "dataset_versioning" {
  bucket = aws_s3_bucket.dataset_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dataset_encryption" {
  bucket = aws_s3_bucket.dataset_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dataset_public_block" {
  bucket = aws_s3_bucket.dataset_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# FSx for Lustre — SCRATCH_2 file system linked to S3 bucket
#
# SCRATCH_2 delivers up to 200 MB/s/TiB throughput and is ideal for
# short-lived, high-performance training workloads.
#
# Minimum capacity: 1,200 GiB (1.2 TB) — the smallest SCRATCH_2 increment.
# -----------------------------------------------------------------------------

resource "aws_fsx_lustre_file_system" "training_fs" {
  storage_capacity            = 1200                              # GiB (minimum for SCRATCH_2)
  subnet_ids                  = [aws_subnet.public.id]
  security_group_ids          = [aws_security_group.cluster_sg.id]
  deployment_type             = "SCRATCH_2"
  import_path                 = "s3://${aws_s3_bucket.dataset_bucket.id}"
  auto_import_policy          = "NEW_CHANGED_DELETED"

  tags = {
    Name    = "${var.project_name}-fsx-lustre"
    Project = var.project_name
  }
}
