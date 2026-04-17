# Avalon Game Workflow Configuration

This directory contains example configuration for running agentscope workflows, specifically for the Avalon game.

## Configuration File

`config.yaml` - Basic configuration file demonstrating how to configure agentscope workflow.

## Key Configuration

The main configuration for agentscope workflow is in `actor_rollout_ref.rollout.agentscope_workflow`:

```yaml
actor_rollout_ref:
  rollout:
    # Specify the agentscope workflow class to use
    # Format: "module.path->WorkflowClassName"
    agentscope_workflow: "games.games.avalon.workflow->AvalonWorkflow"
```

## Workflow Base Class

All workflow classes should inherit from `BaseAgentscopeWorkflow` (defined in `agentevolver.utils.agentscope_utils`):

```python
from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.trajectory import Trajectory

class AvalonWorkflow(BaseAgentscopeWorkflow):
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

## Usage

Run with:

```bash
python launcher.py --config-path examples/game/avalon --config-name config
```

