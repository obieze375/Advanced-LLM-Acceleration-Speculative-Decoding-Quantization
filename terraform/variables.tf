variable "region" {
  description = "Nebius region (H100 SXM is available in eu-north1)."
  type        = string
  default     = "eu-north1"
}

variable "project_id" {
  description = "Nebius project ID from the console (e.g. project-e00...)."
  type        = string
}

variable "subnet_name" {
  description = "VPC subnet name in the console (used to look up subnet_id if subnet_id unset)."
  type        = string
  default     = "default-subnet"
}

variable "subnet_id" {
  description = "VPC subnet ID. If empty, set manually after: nebius vpc subnet list."
  type        = string
  default     = ""
}

variable "instance_name" {
  description = "Name for the GPU VM in Nebius Compute."
  type        = string
  default     = "mlops-assignment-h100"
}

variable "gpu_platform" {
  description = "GPU platform preset family."
  type        = string
  default     = "gpu-h100-sxm"
}

variable "gpu_preset" {
  description = "GPU + CPU + RAM preset."
  type        = string
  default     = "1gpu-16vcpu-200gb"
}

variable "boot_disk_size_gib" {
  description = "Boot disk size in GiB."
  type        = number
  default     = 200
}

variable "boot_disk_image_family" {
  description = "Nebius boot image family."
  type        = string
  default     = "ubuntu24.04-cuda13.0"
}

variable "ssh_username" {
  description = "Linux user created by cloud-init (for your SSH profile reference)."
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "SSH public key contents (paste full key line). Required for SSH access."
  type        = string
  default     = ""
}

variable "key_name" {
  description = "Label for your Nebius SSH key profile (documentation only)."
  type        = string
  default     = ""
}

variable "repo_url" {
  description = "Optional git repo URL to clone on first boot."
  type        = string
  default     = ""
}
