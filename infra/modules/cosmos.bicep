@description('Cosmos DB account name.')
param accountName string

@description('Cosmos DB account location (must match existing region).')
param location string

@description('Cosmos DB SQL database name.')
param sqlDatabaseName string

@description('Cosmos DB SQL container name.')
param containerName string

@description('Whether to manage (create/update) the SQL container via Bicep. Set to false to treat it as existing (avoids immutable schema update failures such as partition key changes).')
param manageContainer bool = true

@description('Partition key paths for the SQL container.')
param partitionKeyPaths array = [
  '/jobId'
]

@description('IP rules allowed to access Cosmos DB (matches existing configuration).')
param ipRules array = []

@description('Optional tag set applied to the Cosmos DB account.')
param tags object = {}

// Cosmos DB account (serverless) - based on exported live configuration.
resource account 'Microsoft.DocumentDB/databaseAccounts@2025-05-01-preview' = {
  name: accountName
  location: location
  kind: 'GlobalDocumentDB'
  // Preserve portal/workload tags while still allowing our standard tags.
  tags: union(tags, {
    defaultExperience: 'Core (SQL)'
    'hidden-cosmos-mmspecial': ''
    'hidden-workload-type': 'Production'
  })
  properties: {
    analyticalStorageConfiguration: {
      schemaType: 'WellDefined'
    }
    backupPolicy: {
      periodicModeProperties: {
        backupIntervalInMinutes: 240
        backupRetentionIntervalInHours: 8
        backupStorageRedundancy: 'Local'
      }
      type: 'Periodic'
    }
    capabilities: []
    capacityMode: 'Serverless'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
      maxIntervalInSeconds: 5
      maxStalenessPrefix: 100
    }
    cors: []
    databaseAccountOfferType: 'Standard'
    defaultIdentity: 'FirstPartyIdentity'
    defaultPriorityLevel: 'High'
    diagnosticLogSettings: {
      enableFullTextQuery: 'None'
    }
    disableKeyBasedMetadataWriteAccess: false
    disableLocalAuth: false
    enableAnalyticalStorage: false
    enableAutomaticFailover: true
    enableBurstCapacity: false
    enableFreeTier: false
    enableMaterializedViews: false
    enableMultipleWriteLocations: false
    enablePartitionMerge: false
    enablePerRegionPerPartitionAutoscale: false
    enablePriorityBasedExecution: false
    ipRules: ipRules
    isVirtualNetworkFilterEnabled: false
    locations: [
      {
        failoverPriority: 0
        isZoneRedundant: true
        locationName: location
      }
    ]
    minimalTlsVersion: 'Tls12'
    networkAclBypass: 'None'
    networkAclBypassResourceIds: []
    publicNetworkAccess: 'Enabled'
    virtualNetworkRules: []
  }
}

resource sqlDb 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2025-05-01-preview' = {
  parent: account
  name: sqlDatabaseName
  properties: {
    resource: {
      id: sqlDatabaseName
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-05-01-preview' = if (manageContainer) {
  parent: sqlDb
  name: containerName
  properties: {
    resource: {
      computedProperties: []
      conflictResolutionPolicy: {
        conflictResolutionPath: '/_ts'
        mode: 'LastWriterWins'
      }
      fullTextPolicy: {
        defaultLanguage: 'en-US'
        fullTextPaths: []
      }
      id: containerName
      indexingPolicy: {
        automatic: true
        excludedPaths: [
          {
            path: '/"_etag"/?'
          }
        ]
        includedPaths: [
          {
            path: '/*'
          }
        ]
        indexingMode: 'consistent'
      }
      partitionKey: {
        kind: 'Hash'
        paths: partitionKeyPaths
        version: 2
      }
      uniqueKeyPolicy: {
        uniqueKeys: []
      }
    }
  }
}

resource existingContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-05-01-preview' existing = if (!manageContainer) {
  parent: sqlDb
  name: containerName
}

output cosmosAccountId string = account.id
output cosmosSqlDatabaseId string = sqlDb.id
output cosmosContainerId string = manageContainer ? container.id : existingContainer.id
