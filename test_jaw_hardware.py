import sys
import time
import json
import os
import serial
import serial.tools.list_ports

print("==========================================")
print("  JAW MOTOR HARDWARE DIAGNOSTIC TEST  ")
print("==========================================")

ports = list(serial.tools.list_ports.comports())
print(f"Available COM ports: {[(p.device, p.description, p.hwid) for p in ports]}")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaw_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        print(f"jaw_config.json: {f.read().strip()}")

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

def test_port(port_name, baud=1000000, motor_id=1):
    print(f"\n--- Testing port {port_name} at {baud} baud ---")
    try:
        ser = serial.Serial(port_name, baud, timeout=0.1)
        print(f"SUCCESS: Opened {port_name}!")
    except Exception as e:
        print(f"FAILED to open {port_name}: {e}")
        return False

    try:
        # Enable torque & set position mode
        write_reg(ser, motor_id, 40, [0]) # torque disable
        time.sleep(0.01)
        write_reg(ser, motor_id, 33, [0]) # mode 0 (position)
        time.sleep(0.01)
        write_reg(ser, motor_id, 40, [1]) # torque enable
        time.sleep(0.01)
        write_reg(ser, motor_id, 46, [1200 & 0xFF, (1200 >> 8) & 0xFF]) # speed

        # Move to position 2288 (open), then 3145 (close)
        print("Sending position 2288 (open)...")
        write_reg(ser, motor_id, 42, [2288 & 0xFF, (2288 >> 8) & 0xFF, 0, 0])
        time.sleep(1.0)

        print("Sending position 3145 (close)...")
        write_reg(ser, motor_id, 42, [3145 & 0xFF, (3145 >> 8) & 0xFF, 0, 0])
        time.sleep(1.0)

        ser.close()
        print(f"Test on {port_name} completed!")
        return True
    except Exception as e:
        print(f"Error during serial write on {port_name}: {e}")
        if ser and ser.is_open:
            ser.close()
        return False

if __name__ == "__main__":
    if not ports:
        print("\n[WARNING] No COM ports detected by Windows! Make sure USB cable is plugged in.")
    else:
        for p in ports:
            # Skip bluetooth ports if desired or test all
            test_port(p.device)
