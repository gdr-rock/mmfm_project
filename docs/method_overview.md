# Method Overview

This project combines text-based planning with frozen predictive video latents.

Core components:

1. Text planner proposes `K` candidate step sequences.
2. Text critic scores linguistic plausibility.
3. Transition bridge `F(ΔS -> Δz)` checks whether textual state changes are consistent with JEPA latent transitions.
4. Goal bridge `G(goal -> z_goal)` grounds language goals in latent space.
5. System-2 selector minimizes:
   `critic_cost + lambda * transition_penalty + mu * goal_distance`.

ASCII pipeline:

```text
Goal text
  |
  v
Candidate planner ----> K candidate plans ----> state-change generator (ΔS)
  |                                                       |
  |                                                       v
  +------------------------ text critic costs         transition bridge F
                                                          |
Video segments --> frozen V-JEPA2 encoder --> z_t, Δz_t --+
                                                          |
Goal text --> goal bridge G --> z_goal --------------------+
                                                          |
                                                          v
                         weighted score + argmin selection -> final plan
```

TODO:

- Replace placeholder bridges with trainable PyTorch modules.
- Add real V-JEPA2 feature extraction backend and batching.
- Extend retrieval benchmark builder with hard negative mining.
