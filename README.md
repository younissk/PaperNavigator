# PaperNavigator

paper navigator helps you discover and analyze academic papers through intelligent search, ranking, and automated report generation.

this was built mainly as a portfolio project and is completely free to use (as long as my azure credits last). if you have any feedback, please go to [this form](https://forms.gle/Nu4sUUeWMSJmCYR28).

if you're interested in working with me, you can reach out via [my website](https://younissk.github.io).

## Overview

PaperNavigator helps researchers discover relevant academic papers by:

1. **Query Profiling**: LLM creates a structured query profile from your research question

![Query Profile](frontend/public/query_profile.png)

2. **Query Augmentation**: LLM augments the query into multiple search queries for broader coverage

![Query Augmentation](frontend/public/query_augmented.png)

3. **Snowball Sampling**: Starting from seed papers, it explores citations and references to find related work

![Snowball Search](frontend/public/snowball.png)

4. **LLM-Based Filtering**: Uses GPT to filter papers for relevance to your research query
5. **ELO Ranking**: Compares papers head-to-head using LLM judgment to create a quality ranking
6. **Report Generation**: Synthesizes findings into a structured research report

## Architecture (Azure)

PaperNavigator runs as a serverless backend on Azure:

![Infrastructure](frontend/public/infra.png)

### Components

- **Frontend** (`frontend/`): React/Vite app for the web interface
- **Azure Functions** (`azure-functions/`): HTTP API + Service Bus worker
- **Core Library** (`papernavigator/`): Shared domain logic (search, ranking, reports)

## Quick Start

### Frontend Dev

```bash
cd frontend
npm install
npm run dev
```

### Azure Functions Local (optional)

If you have Azure Functions Core Tools installed:

```bash
cd azure-functions
func start
```

## Project Structure

```
PaperNavigator/
├── azure-functions/      # Azure Functions entrypoint + modules
├── frontend/             # React/Vite frontend
├── papernavigator/       # Core library (domain logic)
│   ├── elo_ranker/       # ELO ranking system
│   └── report/           # Report generation
├── tests/                # Test suites
└── results/              # Example outputs (historical)
```

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for Python package management
- Node.js 18+ (for frontend)
- Azure Functions Core Tools (optional, for local Functions)

### Setup

```bash
# Install Python dependencies (including test/dev extras)
uv sync --extra test --extra dev

# Install pre-commit hooks
pre-commit install

# Install frontend dependencies
cd frontend && npm install
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=papernavigator
```

### Code Quality

```bash
# Format code
uv run ruff format papernavigator tests azure-functions

# Lint
uv run ruff check papernavigator tests azure-functions

# Type check
uv run pyright
```

## Deployment

- **Infra (backend)**: Bicep + GitHub Actions (`infra/`, `.github/workflows/azure-infra-deploy.yml`)
- **Backend code**: GitHub Actions deploys Azure Functions (`.github/workflows/azure-functions-deploy.yml`)

## Configuration

### Environment Variables (Azure Functions)

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM operations | Yes (or Key Vault) |
| `AZURE_COSMOS_ENDPOINT` | Cosmos DB endpoint | Yes |
| `AZURE_COSMOS_KEY` | Cosmos DB key | Yes |
| `AZURE_COSMOS_DATABASE` | Cosmos DB database name | Yes |
| `AZURE_COSMOS_CONTAINER` | Cosmos DB container name | Yes |
| `AZURE_SERVICE_BUS_CONNECTION_STRING` | Service Bus connection string | Yes |
| `AZURE_SERVICE_BUS_QUEUE_NAME` | Service Bus queue name | Yes |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob storage connection string | Yes |
| `AZURE_RESULTS_CONTAINER` | Blob container for results | Yes |
| `AZURE_RESULTS_PREFIX` | Blob prefix for results | No |
| `AZURE_KEY_VAULT_URL` | Key Vault URL (optional) | No |
| `OPENAI_API_KEY_SECRET_NAME` | Key Vault secret name (optional) | No |
| `PAPERPILOT_FALLBACK_SEED_COUNT` | When strict filtering yields 0 papers, use this many OpenAlex fallback seeds (default: 8) | No |
| `JOB_QUEUED_SECONDS` | Seconds before queued-job watchdog runs the pipeline directly (default: 20) | No |
| `JOB_STALE_MINUTES` | Minutes before stale-job watchdog fails a stuck running job (default: 30) | No |

## Infrastructure as Code (recommended)

The backend Azure infrastructure is being migrated to **Bicep** under `infra/` so the repo becomes the source of truth (PR-reviewed), reducing portal drift.

- Start here: `infra/README.md`
- Resource Group (prod): `PaperPilot`

## API Endpoints

- `POST /api/pipeline` - Start a pipeline job
- `GET /api/pipeline/{job_id}` - Get job status
- `GET /api/jobs/{job_id}` - Raw job status
- `GET /api/jobs/{job_id}/events` - Job event log
- `GET /api/results` - List completed queries
- `GET /api/results/{query}/all` - Get all results for a query

## License

See [LICENSE](LICENSE) — use freely, this is a portfolio project.

## Feedback

Found a bug or have a feature idea? [Submit feedback here](https://forms.gle/Nu4sUUeWMSJmCYR28).
