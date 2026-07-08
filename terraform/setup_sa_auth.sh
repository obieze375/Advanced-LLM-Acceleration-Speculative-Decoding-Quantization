#!/usr/bin/env bash
# One-time Nebius service-account setup for Terraform (Option B).
# Usage:
#   ./setup_sa_auth.sh          # create SA + key, write ~/.nebius/terraform-auth.env
#   source ~/.nebius/terraform-auth.env
#   terraform apply

set -euo pipefail

SA_NAME="${NEBIUS_TERRAFORM_SA_NAME:-terraform-sa}"
AUTH_DIR="${HOME}/.nebius/authkey"
ENV_FILE="${HOME}/.nebius/terraform-auth.env"
PRIVATE_KEY="${AUTH_DIR}/private.pem"
PUBLIC_KEY="${AUTH_DIR}/public.pem"

if ! command -v nebius >/dev/null 2>&1; then
  echo "Install Nebius CLI first: curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash"
  exit 1
fi

if ! nebius profile list >/dev/null 2>&1; then
  echo "Configure Nebius CLI first: nebius profile create"
  exit 1
fi

mkdir -p "${AUTH_DIR}"

echo "==> Resolve or create service account: ${SA_NAME}"
SA_ID=$(nebius iam service-account get-by-name --name "${SA_NAME}" --format json 2>/dev/null | jq -r '.metadata.id' || true)
if [ -z "${SA_ID}" ] || [ "${SA_ID}" = "null" ]; then
  SA_ID=$(nebius iam service-account create --name "${SA_NAME}" --format json | jq -r '.metadata.id')
  echo "Created service account: ${SA_ID}"
else
  echo "Using existing service account: ${SA_ID}"
fi

echo "==> Ensure service account is in editors group"
TENANT_ID=$(nebius iam tenant list --format json | jq -r '.items[0].metadata.id')
EDITORS_GROUP_ID=$(nebius iam group get-by-name --name editors --parent-id "${TENANT_ID}" --format json | jq -r '.metadata.id')
nebius iam group-membership create --parent-id "${EDITORS_GROUP_ID}" --member-id "${SA_ID}" >/dev/null 2>&1 || true

if [ ! -f "${PRIVATE_KEY}" ]; then
  echo "==> Generate RSA key pair"
  openssl genrsa -out "${PRIVATE_KEY}" 4096
  openssl rsa -in "${PRIVATE_KEY}" -outform PEM -pubout -out "${PUBLIC_KEY}"
  chmod 600 "${PRIVATE_KEY}"
else
  echo "==> Reusing existing private key at ${PRIVATE_KEY}"
  [ -f "${PUBLIC_KEY}" ] || openssl rsa -in "${PRIVATE_KEY}" -outform PEM -pubout -out "${PUBLIC_KEY}"
fi

echo "==> Upload public key to Nebius (new authorized key)"
AUTHKEY_PUBLIC_ID=$(nebius iam auth-public-key create \
  --account-service-account-id "${SA_ID}" \
  --data "$(cat "${PUBLIC_KEY}")" \
  --format json | jq -r '.metadata.id')

cat > "${ENV_FILE}" <<EOF
# Source before terraform:  source ${ENV_FILE}
export SA_ID=${SA_ID}
export AUTHKEY_PUBLIC_ID=${AUTHKEY_PUBLIC_ID}
export AUTHKEY_PRIVATE_PATH=${PRIVATE_KEY}
EOF
chmod 600 "${ENV_FILE}"

echo ""
echo "Wrote ${ENV_FILE}"
echo "Run:"
echo "  source ${ENV_FILE}"
echo "  cd terraform && terraform apply"
