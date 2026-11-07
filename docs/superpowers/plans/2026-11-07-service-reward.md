# Service Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable service-success, movement-cost, and wait-time reward shaping.

**Architecture:** Track an MCS completion event across environment steps, then add weighted shaping terms before existing normalization.

**Spec:** `docs/superpowers/specs/2026-11-07-service-reward-design.md`

- [ ] Add failing unit tests for persisted completion reward and configured shaping.
- [ ] Add MCS event/cost state and configurable reward terms.
- [ ] Verify full test suite and rerun a short MLP/GNN comparison.
