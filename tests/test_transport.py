import pytest

from okapi.mount.transport import SerialTransport, TransportTimeoutError


@pytest.fixture
def loop():
    t = SerialTransport("loop://")  # pyserial's loopback: what we write comes straight back
    yield t
    t.close()


def test_read_up_to_the_hash(loop):
    loop.write(b"12AB0500,40000500#")
    assert loop.read_reply(1.0) == b"12AB0500,40000500#"


def test_a_binary_reply_containing_the_hash_byte_is_read_by_length(loop):
    reply = bytes([35, 50, 41, 0, 35, 20, 17, 1]) + b"#"  # latitude 35 is ord('#')
    loop.write(reply)
    assert loop.read_reply(1.0, length=9) == reply


def test_reading_to_the_hash_would_have_truncated_it(loop):
    loop.write(bytes([35, 50, 41, 0, 35, 20, 17, 1]) + b"#")
    assert loop.read_reply(1.0) == b"#"  # why replies with binary payloads use length=


def test_timeout_when_nothing_arrives(loop):
    with pytest.raises(TransportTimeoutError):
        loop.read_reply(0.1)


def test_timeout_when_the_reply_is_short(loop):
    loop.write(b"abc")  # no terminator
    with pytest.raises(TransportTimeoutError):
        loop.read_reply(0.1)


def test_timeout_when_a_fixed_length_reply_is_short(loop):
    loop.write(b"ab#")
    with pytest.raises(TransportTimeoutError):
        loop.read_reply(0.1, length=9)


def test_stale_bytes_are_discarded_before_each_command(loop):
    loop.write(b"old#")  # a reply we never read
    loop.write(b"new#")
    assert loop.read_reply(1.0) == b"new#"
