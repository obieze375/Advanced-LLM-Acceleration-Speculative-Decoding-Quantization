# Nebius H100 Terraform

Provisions a single **NVIDIA H100 80GB** VM (`gpu-h100-sxm` / `1gpu-16vcpu-200gb`) for the MLOps assignment.

Authentication uses **Option B: Nebius service account** (see `providers.tf`).

## Prerequisites

1. [Nebius CLI](https://docs.nebius.com/cli/quickstart) installed and configured (`nebius profile create`).
2. [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5.
3. Your existing **project ID** and **subnet ID** in `eu-north1`.

## Step 1 — Service account auth (one-time)

From the repo root:

```bash
chmod +x terraform/setup_sa_auth.sh
./terraform/setup_sa_auth.sh
```

This creates (or reuses) a `terraform-sa` service account, generates a key pair, and writes:

`~/.nebius/terraform-auth.env`

Before every Terraform run, load those credentials:

```bash
source ~/.nebius/terraform-auth.env
```

That exports three variables used by `providers.tf`:

| Env var | Meaning |
|---------|---------|
| `SA_ID` | Service account ID |
| `AUTHKEY_PUBLIC_ID` | Authorized public key ID |
| `AUTHKEY_PRIVATE_PATH` | Path to private PEM file |

**Do not commit** the private key or `terraform-auth.env`.

### Manual setup (equivalent)

```bash
export SA_ID=$(nebius iam service-account create --name terraform-sa --format json | jq -r '.metadata.id')
export TENANT_ID=$(nebius iam tenant list --format json | jq -r '.items[0].metadata.id')
export EDITORS_GROUP_ID=$(nebius iam group get-by-name --name editors --parent-id "$TENANT_ID" --format json | jq -r '.metadata.id')
nebius iam group-membership create --parent-id "$EDITORS_GROUP_ID" --member-id "$SA_ID"
mkdir -p ~/.nebius/authkey
export AUTHKEY_PRIVATE_PATH=~/.nebius/authkey/private.pem
openssl genrsa -out "$AUTHKEY_PRIVATE_PATH" 4096
openssl rsa -in "$AUTHKEY_PRIVATE_PATH" -outform PEM -pubout -out ~/.nebius/authkey/public.pem
export AUTHKEY_PUBLIC_ID=$(nebius iam auth-public-key create \
  --account-service-account-id "$SA_ID" \
  --data "$(cat ~/.nebius/authkey/public.pem)" \
  --format json | jq -r '.metadata.id')
```

## Step 2 — Project config (`terraform.tfvars`)

Your **existing project** goes here — not in the provider auth block.

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
project_id    = "project-XXXXXXXXXXXXXXXX"   # your existing project
subnet_id     = "subnet-XXXXXXXXXXXXXXXX"
ssh_public_key = "ssh-ed25519 AAAA..."

gpu_platform = "gpu-h100-sxm"
gpu_preset   = "1gpu-16vcpu-200gb"
```

Get IDs:

```bash
nebius project list
nebius vpc subnet list --parent-id <project_id>
```

## Step 3 — Apply

```bash
source ~/.nebius/terraform-auth.env
cd terraform
terraform init
terraform validate
terraform plan
terraform apply
```

SSH:

```bash
terraform output ssh_hint
```

Port-forward for the assignment:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 \
    -L 8000:localhost:8000 -L 8001:localhost:8001 ubuntu@<public-ip>
```

## Destroy

```bash
source ~/.nebius/terraform-auth.env
cd terraform
terraform destroy
```
