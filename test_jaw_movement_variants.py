import time
import serial

def checksum(data):
    return (~sum(data)) & 0xFF

def write_reg(ser, mid, reg, data_bytes):
    body = [0x03, reg] + list(data_bytes)
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.01)

def read_pos(ser, mid=1):
    body = [0x02, 56, 2]
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.02)
    resp = ser.read(20)
    start = resp.find(b"\xff\xff")
    if start >= 0 and len(resp) >= start + 6:
        frame = resp[start:]
        l = frame[3]
        if len(frame) >= 4 + l:
            params = frame[5 : 5 + max(0, l - 2)]
            if len(params) == 2:
                return params[0] | (params[1] << 8)
    return None

def test():
    ser = serial.Serial("COM5", 1000000, timeout=0.05)
    print("Initial Position:", read_pos(ser))

    # Enable torque
    write_reg(ser, 1, 40, [0]) # Torque off
    time.sleep(0.02)
    write_reg(ser, 1, 33, [0]) # Mode 0 (Position mode)
    time.sleep(0.02)
    write_reg(ser, 1, 40, [1]) # Torque on
    time.sleep(0.02)
    write_reg(ser, 1, 46, [1200 & 0xFF, (1200 >> 8) & 0xFF]) # Speed = 1200
    time.sleep(0.02)

    # Test 1: 2-byte goal position write to reg 42
    print("\n--- Test 1: Writing 2 bytes to Reg 42 (target pos 2300) ---")
    write_reg(ser, 1, 42, [2300 & 0xFF, (2300 >> 8) & 0xFF])
    time.sleep(1.0)
    p1 = read_pos(ser)
    print("Position after Test 1:", p1)

    # Test 2: 4-byte write with Goal Time = 0
    print("\n--- Test 2: Writing 4 bytes [pos_L, pos_H, 0, 0] (target pos 2700) ---")
    write_reg(ser, 1, 42, [2700 & 0xFF, (2700 >> 8) & 0xFF, 0, 0])
    time.sleep(1.0)
    p2 = read_pos(ser)
    print("Position after Test 2:", p2)

    # Test 3: 4-byte write with Goal Time = 200ms
    print("\n--- Test 3: Writing 4 bytes [pos_L, pos_H, 200, 0] (target pos 2300) ---")
    write_reg(ser, 1, 42, [2300 & 0xFF, (2300 >> 8) & 0xFF, 200 & 0xFF, (200 >> 8) & 0xFF])
    time.sleep(1.0)
    p3 = read_pos(ser)
    print("Position after Test 3:", p3)

    # Return to home
    print("\n--- Returning to 3137 ---")
    write_reg(ser, 1, 42, [3137 & 0xFF, (3137 >> 8) & 0xFF])
    time.sleep(1.0)
    print("Final Position:", read_pos(ser))

    ser.close()

if __name__ == "__main__":
    test()
