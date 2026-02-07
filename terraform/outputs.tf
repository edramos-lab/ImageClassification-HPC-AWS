# =============================================================================
# outputs.tf — Key resource references printed after terraform apply
# =============================================================================

output "gpu_node_public_ip" {
  description = "Public IP address of the GPU training node (use for SSH)"
  value       = aws_instance.gpu_node.public_ip
}

output "fsx_mount_command" {
  description = "Manual mount command (fallback if user_data didn't run)"
  value       = "sudo mount -t lustre -o noatime,flock ${aws_fsx_lustre_file_system.training_fs.dns_name}@tcp:/${aws_fsx_lustre_file_system.training_fs.mount_name} /home/ubuntu/data"
}

output "s3_bucket_id" {
  description = "S3 bucket ID where datasets are stored"
  value       = aws_s3_bucket.dataset_bucket.id
}

output "fsx_file_system_id" {
  description = "FSx for Lustre file system ID"
  value       = aws_fsx_lustre_file_system.training_fs.id
}

output "ami_id_used" {
  description = "Deep Learning AMI ID selected for the instance"
  value       = data.aws_ami.deep_learning.id
}
