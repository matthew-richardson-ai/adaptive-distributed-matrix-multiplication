import numpy as np

from Aggregator.agg_server import AggregatorState, JobState


def test_job_state_rejects_duplicate_task_result():
    job = JobState(
        job_id="j1",
        output_shape=(2, 2),
        original_shape=(2, 2),
        expected_tasks=1,
        block_size=2,
    )
    partial = np.ones((2, 2), dtype=np.float64)

    assert job.add_result("task-1", 0, 0, partial) is True
    assert job.add_result("task-1", 0, 0, partial) is False
    assert np.allclose(job.result, partial)


def test_zero_expected_tasks_is_immediately_complete():
    job = JobState(
        job_id="all-pruned",
        output_shape=(4, 4),
        original_shape=(3, 3),
        expected_tasks=0,
        block_size=2,
    )
    assert job.complete
    assert np.allclose(job.trimmed_result(), np.zeros((3, 3)))


def test_aggregator_state_creates_job():
    state = AggregatorState()
    job = state.create_job(
        {
            "job_id": "abc",
            "output_shape": [4, 4],
            "original_shape": [3, 3],
            "expected_tasks": 8,
            "block_size": 2,
            "total_candidate_tasks": 10,
            "pruned_tasks": 2,
        }
    )

    assert state.get_job("abc") is job
    assert job.total_candidate_tasks == 10
    assert job.pruned_tasks == 2
