# GPU VM for the MLOps assignment (default: 1× H100 80GB on gpu-h100-sxm).

resource "nebius_compute_v1_disk" "boot" {
  name           = "${var.instance_name}-boot"
  parent_id      = var.project_id
  size_gibibytes = var.boot_disk_size_gib
  type           = "NETWORK_SSD"
  source_image_family = {
    image_family = var.boot_disk_image_family
  }
  block_size_bytes = 4096
}

resource "nebius_compute_v1_instance" "gpu" {
  name      = var.instance_name
  parent_id = var.project_id

  resources = {
    platform = var.gpu_platform
    preset   = var.gpu_preset
  }

  boot_disk = {
    existing_disk = {
      id = nebius_compute_v1_disk.boot.id
    }
    attach_mode = "READ_WRITE"
  }

  cloud_init_user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    ssh_public_key = var.ssh_public_key
    repo_url       = var.repo_url
    ssh_username   = var.ssh_username
  })

  network_interfaces = [
    {
      name       = "eth0"
      ip_address = {}
      public_ip_address = {
        static = true
      }
      subnet_id = var.subnet_id
    }
  ]
}

output "region" {
  value = var.region
}

output "instance_id" {
  value = nebius_compute_v1_instance.gpu.id
}

output "gpu_platform" {
  value = var.gpu_platform
}

output "gpu_preset" {
  value = var.gpu_preset
}

output "public_ip" {
  description = "SSH target for the assignment VM (bare IP, no CIDR suffix)."
  value       = split("/", try(nebius_compute_v1_instance.gpu.status.network_interfaces[0].public_ip_address.address, ""))[0]
}

output "ssh_hint" {
  value = "ssh ${var.ssh_username}@${split("/", try(nebius_compute_v1_instance.gpu.status.network_interfaces[0].public_ip_address.address, "<public-ip>"))[0]}"
}
