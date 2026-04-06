from __future__ import annotations

import unittest

from aca.contracts import (
    ChallengerCritique,
    MasterAnalyzeBrief,
    MasterFinalPlan,
    MasterImplementationPlan,
    ParallelStepGroup,
    WorkerTaskResult,
    WorkerTaskSpec,
)


class StructuredContractSchemaTests(unittest.TestCase):
    def test_contract_schemas_forbid_additional_properties(self) -> None:
        for model_cls in (
            WorkerTaskSpec,
            ParallelStepGroup,
            MasterImplementationPlan,
            ChallengerCritique,
            MasterFinalPlan,
            MasterAnalyzeBrief,
            WorkerTaskResult,
        ):
            schema = model_cls.model_json_schema()
            self.assertFalse(schema.get("additionalProperties", True), model_cls.__name__)

    def test_worker_task_result_status_is_restricted_enum(self) -> None:
        schema = WorkerTaskResult.model_json_schema()
        self.assertEqual(schema["properties"]["status"]["enum"], ["completed", "failed"])


if __name__ == "__main__":
    unittest.main()
