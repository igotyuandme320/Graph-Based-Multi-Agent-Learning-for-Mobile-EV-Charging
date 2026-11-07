# Graph-Based Multi-Agent Learning for Mobile EV Charging

Application and repository copy draft. The public repository URL and upstream attribution still need to be supplied before publication. This document does not claim that the project has already been publicly released.

## Application form: proudest open-source project

One project I am particularly proud of is Graph-Based Multi-Agent Learning for Mobile EV Charging, an extension of an existing reinforcement-learning simulator. I explored how mobile charging stations and electric vehicles could make better spatial decisions under limited resources and local observations. Building on a DQN baseline, I added spatial graph attention over candidate locations, experience replay, exploration scheduling, and configurable reward shaping, together with reproducible training and evaluation workflows.

The most valuable part was investigating results that challenged my expectations. The graph policy did not consistently outperform the simpler baseline, which pushed me to examine training quality, reward design, and evaluation rather than assume that a more expressive model would be better. This project strengthened my interest in how agents learn useful spatial representations and translate local information into effective sequential decisions.

## Short repository description

A reproducible PyTorch extension of an existing EV/mobile-charging-station simulator, exploring spatial graph-attention DQN policies, experience replay, and reward shaping for decentralized scheduling.

## Technical contribution summary

- Extended an existing simulator; the traffic environment, trajectory data, and original DQN baseline predate this work.
- Added a candidate-location graph Q-network with spatial k-nearest-neighbor edges and multi-head attention. EV and MCS entities are not separate graph node types in the current implementation.
- Added bounded experience replay, exploration decay, terminal-target handling, and numerical-stability checks. Sampled transitions currently receive sequential optimizer updates rather than one averaged minibatch update.
- Added configurable service-revenue, movement-cost, and waiting reward terms. MCS service revenue is assigned to the dispatch action that initiated the task period through a duration-aware SMDP replay transition; the effect of reward normalization still requires ablation.
- Added Conda setup, CLI training/evaluation, checkpoint handling, automated tests, and a shared-budget multi-seed benchmark that records episode-level, seed-run, and policy-level results.

## Evidence and limitations

- Earlier comparisons used a small high-demand setting and nonuniform training budgets; they are exploratory results, not evidence of a general performance advantage. The repository now includes a shared-budget multi-seed benchmark protocol, which must be rerun before making comparative claims.
- Evaluation now distinguishes requested, assigned, completed, failed, unserved, and still-in-progress charging requests. `charged_proportion` is retained as a compatibility alias for verified completion rate.
- Waiting-time and extra-distance averages for completed, assigned, and failed requests are reported separately; these metrics must be interpreted alongside service coverage.
- The graph uses spatial proximity between candidate locations, not verified road-connectivity edges.
- HER, a learned world model, language conditioning, and a full EV–MCS–road heterogeneous graph are not implemented.

## Before submitting as an open-source project

- Add the actual public repository URL.
- Identify and credit the upstream simulator and distinguish inherited code from extensions.
- Establish the permitted redistribution terms for code, datasets, and checkpoints.
- Publish exact experiment configurations, seeds, and commands alongside the results.
- Check that the applicant can explain the implementation, development assistance, and limitations accurately.

## Possible next research question

When a charging request is not served, was it infeasible under the available resources, or did the learned policy miss a feasible sequence of decisions? A failure taxonomy and feasibility checks would help separate resource limitations from policy errors before introducing a more complex architecture.
