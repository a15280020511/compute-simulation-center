# Mesa Agent-Based Simulation Guide

## Boundary

`agent_based_simulation` is the only Mesa-backed compute operation. It loads the exact pinned `mesa==3.5.1` environment only for that operation. The core compute requirements remain unchanged.

The operation accepts JSON parameters for fixed models only. It never accepts Python source, class names, module paths, arbitrary functions, visualization servers, runtime plugins, URLs, or network access.

Hard limits:

- agents: 5,000;
- steps: 1,000;
- options/resources: 20;
- generated network edges: 100,000;
- deterministic seed: 0 through 2^32-1.

## Mode selection

### `heterogeneous_worker_choice`

Use when individual workers, drivers, riders, customers, or firms choose among alternatives while differing in preferences, learning from rewards, paying switching costs, and experiencing congestion.

```json
{
  "task_id": "mesa-worker-choice-001",
  "operation": "agent_based_simulation",
  "inputs": {
    "mode": "heterogeneous_worker_choice",
    "agent_count": 1000,
    "steps": 200,
    "seed": 20260728,
    "learning_rate": 0.2,
    "choice_sensitivity": 2.0,
    "switching_cost": 1.5,
    "preference_standard_deviation": 0.8,
    "reward_standard_deviation": 1.0,
    "options": [
      {
        "name": "zone_a",
        "base_reward": 25,
        "cost": 8,
        "risk_cost": 1,
        "capacity": 400,
        "congestion_penalty": 10
      },
      {
        "name": "zone_b",
        "base_reward": 22,
        "cost": 5,
        "risk_cost": 0.5,
        "capacity": 700,
        "congestion_penalty": 5
      }
    ]
  }
}
```

### `network_contagion`

Use for bounded adoption, information diffusion, behavior spread, or recovery on a generated undirected contact network. The model uses synchronous state updates and does not download or accept an external graph file.

```json
{
  "task_id": "mesa-contagion-001",
  "operation": "agent_based_simulation",
  "inputs": {
    "mode": "network_contagion",
    "agent_count": 2000,
    "steps": 100,
    "seed": 7,
    "average_degree": 8,
    "initial_adoption_rate": 0.05,
    "threshold_mean": 0.35,
    "threshold_standard_deviation": 0.1,
    "external_influence": 0.02,
    "recovery_rate": 0.01
  }
}
```

### `resource_competition`

Use when heterogeneous agents repeatedly choose among renewable or exhaustible resources, learn from returns, and compete for limited stock.

```json
{
  "task_id": "mesa-resource-001",
  "operation": "agent_based_simulation",
  "inputs": {
    "mode": "resource_competition",
    "agent_count": 800,
    "steps": 150,
    "seed": 11,
    "demand_mean": 1.0,
    "demand_standard_deviation": 0.2,
    "learning_rate": 0.2,
    "exploration_rate": 0.05,
    "choice_sensitivity": 2.0,
    "resources": [
      {
        "name": "resource_a",
        "initial_stock": 1000,
        "capacity": 1000,
        "regeneration": 200,
        "unit_value": 2
      },
      {
        "name": "resource_b",
        "initial_stock": 700,
        "capacity": 700,
        "regeneration": 100,
        "unit_value": 3
      }
    ]
  }
}
```

## GPTs selection rule

Before creating a `[compute]` Issue, GPTs must call `getComputeToolCatalog`, decode `compute-capabilities.json`, and select the smallest sufficient operation. Use `agent_evolution` for aggregate strategy-share dynamics. Use `agent_based_simulation` only when individual heterogeneity, local interaction, learning, switching, congestion, or resource competition materially affects the result.

Then call `getComputeTicketSchema`, validate the ticket shape, and create one `[compute]` Issue. Mesa does not communicate with the API center or expert center; GPTs remains the only cross-center relay.
