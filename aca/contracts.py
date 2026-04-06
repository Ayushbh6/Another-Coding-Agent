from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictStructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerTaskSpec(StrictStructuredModel):
    step_id: str
    title: str
    instructions: str
    allowed_mutation: bool = False
    acceptance_checks: list[str] = Field(default_factory=list)


class ParallelStepGroup(StrictStructuredModel):
    group_id: str
    steps: list[WorkerTaskSpec] = Field(default_factory=list)


class MasterImplementationPlan(StrictStructuredModel):
    task_title: str
    goal: str
    todo: list[str] = Field(default_factory=list)
    sequential_steps: list[WorkerTaskSpec] = Field(default_factory=list)
    parallel_step_groups: list[ParallelStepGroup] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    worker_global_instructions: str


class ChallengerCritique(StrictStructuredModel):
    summary: str
    risks: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    bad_assumptions: list[str] = Field(default_factory=list)
    recommended_plan_changes: list[str] = Field(default_factory=list)


class MasterFinalPlan(MasterImplementationPlan):
    pass


class MasterAnalyzeBrief(StrictStructuredModel):
    task_title: str
    questions_to_answer: list[str] = Field(default_factory=list)
    worker_brief: str
    expected_answer_shape: list[str] = Field(default_factory=list)


class WorkerTaskResult(StrictStructuredModel):
    status: Literal["completed", "failed"]
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
