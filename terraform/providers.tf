# Option B: authenticate with a Nebius service account (recommended).
# Before terraform apply, source credentials:
#   source ~/.nebius/terraform-auth.env
# (created by ./setup_sa_auth.sh)
provider "nebius" {
  service_account = {
    private_key_file_env = "AUTHKEY_PRIVATE_PATH"
    public_key_id_env    = "AUTHKEY_PUBLIC_ID"
    account_id_env       = "SA_ID"
  }
}
