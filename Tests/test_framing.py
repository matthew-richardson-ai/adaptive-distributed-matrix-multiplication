import socket
import threading

import pytest

from Protocol.constants import MSG_HEARTBEAT
from Protocol.framing import ProtocolError, recv_message, send_message


def test_message_round_trip():
    left, right = socket.socketpair()
    try:
        send_message(left, MSG_HEARTBEAT, {"worker_id": "w1"}, b"abc")
        header, payload = recv_message(right)
        assert header["type"] == MSG_HEARTBEAT
        assert header["worker_id"] == "w1"
        assert payload == b"abc"
    finally:
        left.close()
        right.close()


def test_multiple_messages_share_connection():
    left, right = socket.socketpair()
    try:
        send_message(left, MSG_HEARTBEAT, {"n": 1})
        send_message(left, MSG_HEARTBEAT, {"n": 2})
        first, _ = recv_message(right)
        second, _ = recv_message(right)
        assert first["n"] == 1
        assert second["n"] == 2
    finally:
        left.close()
        right.close()


def test_closed_socket_before_frame_raises():
    left, right = socket.socketpair()
    left.close()
    try:
        with pytest.raises(ConnectionError):
            recv_message(right)
    finally:
        right.close()
