import os
import time
import uuid


def generate_uuid7() -> uuid.UUID:
    """Generate RFC 9562 compliant UUIDv7 (time-ordered 128-bit identifier)."""
    # 48-bit timestamp in milliseconds
    ms = int(time.time() * 1000)
    # 74 bits of randomness
    rand = os.urandom(10)

    b = bytearray(16)
    b[0:6] = ms.to_bytes(6, byteorder="big")
    b[6] = 0x70 | (rand[0] & 0x0F)  # version 7
    b[7] = rand[1]
    b[8] = 0x80 | (rand[2] & 0x3F)  # variant RFC 4122/9562
    b[9:16] = rand[3:10]

    return uuid.UUID(bytes=bytes(b))


def generate_uuid7_str() -> str:
    """Generate RFC 9562 compliant UUIDv7 as a canonical string."""
    return str(generate_uuid7())
