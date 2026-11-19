# TCP Protocol Specification

## Goal

Define a minimal, deterministic application-layer protocol for communication between Ingress, Worker, and Aggregator services.

TCP is a byte stream and does not preserve message boundaries. Every message therefore uses an explicit header and payload length.

## Message Frame

Recommended frame:

```text
+------------------+
| Magic            |
| Version          |
| Message Type     |
| Header Length    |
| Payload Length   |
| Task ID          |
| Metadata         |
+------------------+
| Payload          |
+------------------+
```

The exact binary layout may change during implementation, but every receiver must know the payload length before reading the payload.

## Required Message Types

Initial baseline:

- TASK_DENSE
- RESULT_RETURN
- HEARTBEAT
- ERROR

Later adaptive extensions may add:

- TASK_A_FACTORED
- TASK_B_FACTORED
- TASK_BOTH_FACTORED

## Dense Task Metadata

A dense multiplication task should include:

- task ID
- output block row `i`
- output block column `j`
- reduction index `k`
- left block rows / columns
- right block rows / columns
- dtype identifier
- payload byte count

Payload order:

```text
[A block bytes][B block bytes]
```

Use contiguous float64 buffers for the first implementation.

## Result Metadata

A result message should include:

- task ID
- output block row `i`
- output block column `j`
- reduction index `k`
- result rows / columns
- dtype
- payload byte count

Payload:

```text
[partial product bytes]
```

## Receive Rules

A receiver must:

1. read exactly the fixed header size
2. verify magic and protocol version
3. validate message type
4. validate declared dimensions
5. validate payload length against configured maximums
6. repeatedly call receive until the full payload is collected
7. decode only after the complete payload is available

Never assume one `recv()` call returns one complete message.

## Error Messages

Error responses should carry:

- task ID when available
- service / worker ID
- error category
- concise human-readable message

Suggested categories:

- INVALID_HEADER
- UNSUPPORTED_VERSION
- UNKNOWN_MESSAGE_TYPE
- INVALID_DIMENSIONS
- PAYLOAD_LENGTH_MISMATCH
- TASK_EXECUTION_FAILURE
- UNKNOWN_TASK
- DUPLICATE_RESULT
- INTERNAL_ERROR

## Connection Strategy

Use persistent TCP connections where practical.

Avoid opening a new TCP connection for every block task because connection setup would distort benchmark measurements.

## Protocol Tests

At minimum test:

- one complete task frame
- one complete result frame
- header split across multiple receives
- payload split across multiple receives
- invalid magic
- unsupported version
- payload length mismatch
- malformed dimensions
- duplicate result handling
