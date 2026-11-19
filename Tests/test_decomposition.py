import numpy as np

from Aggregator.agg_server import JobState
from Ingress.decomposition import count_block_tasks, iter_block_tasks
from Worker.kernels import dense_multiply


def test_task_count_for_two_by_two_block_grid():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b = np.eye(4, dtype=np.float64)
    assert count_block_tasks(a, b, block_size=2) == 8


def test_block_tasks_reconstruct_serial_result():
    a = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    b = np.arange(17, 33, dtype=np.float64).reshape(4, 4)

    tasks = list(iter_block_tasks(a, b, block_size=2, job_id="job"))
    job = JobState(
        job_id="job",
        output_shape=(4, 4),
        original_shape=(4, 4),
        expected_tasks=len(tasks),
        block_size=2,
    )

    for task in tasks:
        partial = dense_multiply(task.a_block, task.b_block)
        assert job.add_result(
            task.task_id,
            task.row_start,
            task.col_start,
            partial,
        )

    assert job.complete
    assert np.allclose(job.trimmed_result(), a @ b)


def test_non_multiple_dimensions_are_padded_and_trimmed():
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    b = np.arange(20, dtype=np.float64).reshape(5, 4)

    tasks = list(iter_block_tasks(a, b, block_size=2, job_id="job-pad"))
    job = JobState(
        job_id="job-pad",
        output_shape=(4, 4),
        original_shape=(3, 4),
        expected_tasks=len(tasks),
        block_size=2,
    )

    for task in tasks:
        job.add_result(
            task.task_id,
            task.row_start,
            task.col_start,
            dense_multiply(task.a_block, task.b_block),
        )

    assert job.trimmed_result().shape == (3, 4)
    assert np.allclose(job.trimmed_result(), a @ b)
