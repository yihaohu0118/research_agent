# Diplomacy Game Workflow Configuration

This directory contains example configuration for running agentscope workflows, specifically for the Diplomacy game.

## Configuration File

`config.yaml` - Basic configuration file demonstrating how to configure agentscope workflow.

## Key Configuration

The main configuration for agentscope workflow is in `actor_rollout_ref.rollout.agentscope_workflow`:

```yaml
actor_rollout_ref:
  rollout:
    # Specify the agentscope workflow class to use
    # Format: "module.path->WorkflowClassName"
    agentscope_workflow: "games.games.diplomacy.workflows.rollout_workflow->DiplomacyWorkflow"
```

## Workflow Base Class

All workflow classes should inherit from `BaseAgentscopeWorkflow` (defined in `agentevolver.utils.agentscope_utils`):

```python
from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.trajectory import Trajectory

class DiplomacyWorkflow(BaseAgentscopeWorkflow):
    def __init__(self, task, llm_chat_fn, model_name, **kwargs):
        super().__init__(task, llm_chat_fn, model_name, **kwargs)
        # Create agents using self.model
        # self.agents = [...]
    
    def execute(self) -> Trajectory:
        # Run the workflow
        # Collect model call history from self.agents
        # Return Trajectory
        pass
```

## Training Tasks

Before training, you need to generate training tasks using:

```bash
python games/games/diplomacy/generate_train_tasks_parquet.py \
    --config games/games/diplomacy/configs/task_config.yaml \
    --output games/games/diplomacy/train_tasks.parquet \
    --num_tasks 100 \
    --train_powers ENGLAND FRANCE
```

## Usage

Run with:

```bash
python launcher.py --config-path examples/game/diplomacy --config-name config
```

Or use the provided shell scripts:

```bash
# For production training
./examples/game/diplomacy/run_train.sh

# For debug training (smaller model, fewer workers)
./examples/game/diplomacy/run_train_debug.sh
```

