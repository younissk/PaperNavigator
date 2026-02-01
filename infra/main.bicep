targetScope = 'resourceGroup'

@description('Location for new resources created by this template.')
param location string = resourceGroup().location

@description('Function App name.')
param functionAppName string

@description('App Service plan (serverfarm) name for the Function App.')
param appServicePlanName string

@description('User-assigned managed identity name for the Function App.')
param userAssignedIdentityName string

@description('Name of the Key Vault to create.')
param keyVaultName string

@description('Cosmos DB account name.')
param cosmosAccountName string

@description('Cosmos DB location (must match existing region).')
param cosmosLocation string

@description('Cosmos DB SQL database name.')
param cosmosSqlDatabaseName string

@description('Cosmos DB SQL container name.')
param cosmosContainerName string

@description('Whether to manage (create/update) the Cosmos SQL container via Bicep. Set to false to treat it as existing (avoids immutable schema update failures such as partition key changes).')
param cosmosManageContainer bool = true

@description('Cosmos DB IP rules allowed to access the account.')
param cosmosIpRules array = []

@description('Service Bus namespace name.')
param serviceBusNamespaceName string

@description('Service Bus queue name.')
param serviceBusQueueName string

@description('Storage account name.')
param storageAccountName string

@description('Blob containers to ensure exist in the storage account.')
param storageBlobContainerNames array = []

@description('Azure Files share used by the Function App content mount (if any).')
param storageFileShareName string = ''

@description('Allowed CORS origins for the Function App.')
param functionCorsAllowedOrigins array = [
  'https://portal.azure.com'
]

@description('Optional custom domain to bind to the Function App (e.g. api.example.com).')
param functionCustomDomainName string = ''

@description('Optional Application Insights resource id to preserve Function App hidden-link tag.')
param appInsightsResourceId string = ''

@description('Optional tag set applied to newly created resources.')
param tags object = {}

// Key Vault (Phase 1)
module keyVault 'modules/keyVault.bicep' = {
  name: 'keyVault'
  params: {
    location: location
    keyVaultName: keyVaultName
    tags: tags
  }
}

// Cosmos DB (Phase 2)
module cosmos 'modules/cosmos.bicep' = {
  name: 'cosmos'
  params: {
    accountName: cosmosAccountName
    location: cosmosLocation
    sqlDatabaseName: cosmosSqlDatabaseName
    containerName: cosmosContainerName
    manageContainer: cosmosManageContainer
    ipRules: cosmosIpRules
    tags: tags
  }
}

// Service Bus (Phase 2)
module serviceBus 'modules/serviceBus.bicep' = {
  name: 'serviceBus'
  params: {
    namespaceName: serviceBusNamespaceName
    location: location
    queueName: serviceBusQueueName
    tags: tags
  }
}

// Storage (Phase 2)
module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    accountName: storageAccountName
    location: location
    blobContainerNames: storageBlobContainerNames
    fileShareName: storageFileShareName
    tags: tags
  }
}

// Function App (Phase 2)
module functionAppFull 'modules/functionAppFull.bicep' = {
  name: 'functionAppFull'
  params: {
    functionAppName: functionAppName
    location: location
    appServicePlanName: appServicePlanName
    userAssignedIdentityName: userAssignedIdentityName
    corsAllowedOrigins: functionCorsAllowedOrigins
    customDomainName: functionCustomDomainName
    appInsightsResourceId: appInsightsResourceId
    tags: tags
  }
}

// NOTE: Key Vault RBAC role assignment ("Key Vault Secrets User") for the
// Function App identity is a one-time setup step requiring elevated permissions
// (Owner or User Access Administrator). It is NOT managed here because the
// GitHub Actions SP lacks roleAssignments/write. See infra/README.md for the
// manual command if deploying from scratch.

output keyVaultUri string = keyVault.outputs.vaultUri
output functionAppNameOut string = functionAppName
output cosmosAccountId string = cosmos.outputs.cosmosAccountId
output serviceBusNamespaceId string = serviceBus.outputs.serviceBusNamespaceId
output storageAccountId string = storage.outputs.storageAccountId
