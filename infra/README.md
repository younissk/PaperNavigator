# Azure Infrastructure (Bicep)

This repo uses **Azure Bicep** to make Azure backend infrastructure **reviewable and reproducible**. The goal is that **PRs are the source of truth**, not ad-hoc portal edits.

## What this manages (Phase 2)

- **Key Vault**: created and used to store secrets (no secrets in git)
- **Cosmos DB**: account (serverless) + SQL database + container
- **Service Bus**: namespace + queue
- **Storage Account**: blob containers + file share for Function App content
- **Function App**: App Service plan (Consumption) + Linux Python Function App + user-assigned identity + web config / CORS
- **Function App app settings**: set to **Key Vault references** via workflow CLI step (no secret values in git)

## Deploy (prod)

Prereqs:
- Azure CLI logged in (`az login`)
- Contributor access to resource group `PaperPilot`

Note: Cosmos DB container partition keys are **immutable**. If the container already exists and the template tries to change the partition key (or other immutable schema), deployments will fail with errors like `Document collection partition key cannot be changed`.

Deploy:

```bash
az deployment group create \
  -g PaperPilot \
  -f infra/main.bicep \
  -p infra/params/prod.bicepparam
```

## Secrets flow (Key Vault references)

This project uses App Service **Key Vault references** for sensitive settings. The pattern is:

- You **store** the secret in Key Vault (one-time / when rotating)
- The Function App setting value becomes a reference like:
  - `@Microsoft.KeyVault(SecretUri=https://<vault>.vault.azure.net/secrets/<name>/<version>)`

Docs: `https://learn.microsoft.com/en-us/azure/app-service/app-service-key-vault-references`

### One-time setup (fresh deployment)

1) Deploy infra to create the Key Vault and Function App identity.
2) **Grant the identity access to read secrets** (requires Owner or User Access Administrator):

```bash
RG="PaperPilot"
UAI_NAME="paperpilot-api-id-97d1"
KV_NAME="paperpilot-kv-prod"

UAI_PRINCIPAL_ID="$(az identity show -g "${RG}" -n "${UAI_NAME}" --query principalId -o tsv)"
KV_ID="$(az keyvault show -g "${RG}" -n "${KV_NAME}" --query id -o tsv)"

az role assignment create \
  --assignee-object-id "${UAI_PRINCIPAL_ID}" \
  --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" \
  --scope "${KV_ID}"
```

3) Populate Key Vault secrets (see below).
4) Re-run the workflow or manually set app settings to Key Vault references.

This repo's default production wiring (Resource Group `PaperPilot`) uses:

- **Function App**: `paperpilot-api`
- **Key Vault**: `paperpilot-kv-prod`
- **User-assigned identity**: `paperpilot-api-id-97d1`

### Secret names (prod)

These Key Vault secrets are expected (names are case-sensitive):

- `OPENAI-API-KEY` → `OPENAI_API_KEY`
- `AZURE-COSMOS-KEY` → `AZURE_COSMOS_KEY`
- `AZURE-SERVICE-BUS-CONNECTION-STRING` → `AZURE_SERVICE_BUS_CONNECTION_STRING`
- `AZURE-WEBJOBS-STORAGE` → `AzureWebJobsStorage`

Note: we intentionally **do not** move `WEBSITE_CONTENTAZUREFILECONNECTIONSTRING` into Key Vault, because it is part of the Azure Files content mount configuration and is riskier to change.

### Useful CLI snippets (prod)

Populate secrets (values are sourced from existing app settings; nothing is committed to git):

```bash
RG="PaperPilot"
FUNC_APP="paperpilot-api"
KV_NAME="paperpilot-kv-prod"

OPENAI_API_KEY="$(az functionapp config appsettings list -g "$RG" -n "$FUNC_APP" --query "[?name=='OPENAI_API_KEY'].value | [0]" -o tsv)"
az keyvault secret set --vault-name "$KV_NAME" --name "OPENAI-API-KEY" --value "$OPENAI_API_KEY" -o none
```

Switch app settings to Key Vault references:

```bash
KV_URI="https://paperpilot-kv-prod.vault.azure.net"
az functionapp config appsettings set -g PaperPilot -n paperpilot-api --settings \
  "OPENAI_API_KEY=@Microsoft.KeyVault(SecretUri=${KV_URI}/secrets/OPENAI-API-KEY)"
```

## Drift policy (important)

- **No portal edits for app settings / infra wiring**.
- If you must do an emergency portal edit, follow up with a PR that updates Bicep/workflows the same day.

## Notes

- The Key Vault RBAC role assignment ("Key Vault Secrets User") is **not** managed by Bicep because the GitHub Actions service principal lacks `Microsoft.Authorization/roleAssignments/write` permission. This is intentional (least-privilege). See "One-time setup" above for fresh deployments.
