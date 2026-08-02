import time
import serial
import serial.tools.list_ports

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

def read_reg(ser, mid, reg, n_bytes):
    body = [0x02, reg, n_bytes]
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.02)
    resp = ser.read(20)
    start = resp.find(b"\xff\xff")
    if start < 0 or len(resp) < start + 6:
        return None
    frame = resp[start:]
    l = frame[3]
    if len(frame) < 4 + l:
        return None
    params = frame[5 : 5 + max(0, l - 2)]
    return params[:n_bytes]

def scan():
    print("==========================================")
    print("  SERVO ID & BAUD RATE SCANNER  ")
    print("==========================================")
    
    ports = [p.device for p in serial.tools.list_ports.comports() if "COM" in p.device]
    if not ports:
        print("No COM ports found!")
        return

    port = "COM5" if "COM5" in ports else ports[0]
    print(f"Scanning on port: {port}")

    bauds = [1000000, 115200, 57600]
    found = []

    for baud in bauds:
        print(f"\nChecking baud rate {baud}...")
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
        except Exception as e:
            print(f"Could not open {port} at {baud}: {e}")
            continue

        for mid in range(1, 16):
            # Try reading position register (reg 56, 2 bytes)
            data = read_reg(ser, mid, 56, 2)
            if data and len(data) == 2:
                pos = data[0] | (data[1] << 8)
                print(f"  ==> FOUND SERVO! Motor ID: {mid}, Present Position: {pos}, Baud: {baud}")
                found.append((mid, baud, pos))
        
        # Try broadcast ID 254 (0xFE) ping
        data_b = read_reg(ser, 0xFE, 56, 2)
        if data_b and len(data_b) == 2:
            pos = data_b[0] | (data_b[1] << 8)
            print(f"  ==> BROADCAST PING SUCCESS! Position: {pos}")

        ser.close()

    if not found:
        print("\n[RESULT] No response received from any motor ID (1-15).")
        print("Possible causes:")
        print("1. Motor power supply (12V/7.4V) is OFF.")
        print("2. Serial GND/TX/RX wiring is reversed or disconnected.")
        print("3. Another process is holding COM5.")
    else:
        print(f"\n[SUCCESS] Detected servos: {found}")

if __name__ == "__main__":
    scan()
