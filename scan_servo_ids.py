"""
scan_servo_ids.py  --  Feetech / SCServo diagnostic scanner
-----------------------------------------------------------
Stage 1 : Loopback test  -- short RX-TX on the USB adapter, confirms
           the CH343 can echo its own bytes. If this fails, the adapter
           itself is the problem.
Stage 2 : Full baud-rate x ID sweep across every standard Feetech baud
           and every valid servo ID (1-253) plus broadcast (0xFE).
Stage 3 : Prints a ranked list of possible causes if nothing is found.
"""
import time
import serial
import serial.tools.list_ports

# -- All standard baud rates the Feetech / SCServo family supports ----------
FEETECH_BAUDS = [1_000_000, 500_000, 250_000, 128_000, 115_200,
                 76_800, 57_600, 38_400, 19_200, 9_600]

# -- Feetech Servo Protocol helpers -----------------------------------------
def checksum(data):
    return (~sum(data)) & 0xFF

def ping_packet(mid):
    """Build a PING instruction packet for motor ID mid."""
    body   = [0x01]           # instruction = PING
    length = len(body) + 1
    pkt    = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    return bytes(pkt)

def read_reg(ser, mid, reg, n_bytes):
    body   = [0x02, reg, n_bytes]
    length = len(body) + 1
    pkt    = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.02)
    resp  = ser.read(32)
    start = resp.find(b"\xff\xff")
    if start < 0 or len(resp) < start + 6:
        return None
    frame = resp[start:]
    l = frame[3]
    if len(frame) < 4 + l:
        return None
    params = frame[5: 5 + max(0, l - 2)]
    return params[:n_bytes]

def try_ping(ser, mid):
    """Send a PING and return True if we get any valid response header."""
    pkt = ping_packet(mid)
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    time.sleep(0.015)
    resp  = ser.read(32)
    start = resp.find(b"\xff\xff")
    return start >= 0 and len(resp) >= start + 6

# -- Stage 1: Loopback test --------------------------------------------------
def loopback_test(port, baud=1_000_000):
    print("\n==========================================")
    print(" STAGE 1 - Adapter loopback test")
    print("==========================================")
    print("  Shorting TX-RX on the CH343 adapter is optional but")
    print("  highly recommended to rule out a broken adapter.")
    print(f"  Opening {port} @ {baud} baud for loopback ...")

    try:
        ser = serial.Serial(port, baud, timeout=0.05)
    except Exception as e:
        print(f"  [FAIL] Cannot open port: {e}")
        return False

    TEST_BYTES = b"\xFF\x55\xAA\x01\xFE"
    ser.reset_input_buffer()
    ser.write(TEST_BYTES)
    ser.flush()
    time.sleep(0.03)
    echo = ser.read(len(TEST_BYTES))
    ser.close()

    if echo == TEST_BYTES:
        print("  [PASS] TX-RX loopback confirmed - adapter is healthy.")
        return True
    elif echo:
        print(f"  [PARTIAL] Got {echo.hex()} instead of {TEST_BYTES.hex()}")
        print("  The adapter may be OK but TX-RX aren't shorted - that's normal")
        print("  if the servo DATA wire is still connected (servo absorbs the bytes).")
        return None
    else:
        print("  [INFO] No echo received.")
        print("  This is EXPECTED if you did NOT short TX-RX.")
        print("  Disconnect the servo DATA wire, jumper TX-RX, then re-run")
        print("  to confirm the adapter is alive.")
        return None

# -- Stage 2: Full sweep -----------------------------------------------------
def full_sweep(port):
    print("\n==========================================")
    print(" STAGE 2 - Full baud x ID sweep")
    print("==========================================")

    found = [] # will store tuples: (mid, baud, pos, model, fw)

    for baud in FEETECH_BAUDS:
        print(f"\n  Trying baud {baud:>9,} ...", end=" ", flush=True)
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
        except Exception as e:
            print(f"open error: {e}")
            continue

        if try_ping(ser, 0xFE):
            print()
            print(f"  *** BROADCAST PING response at baud {baud} ***")
            for mid in range(1, 254):
                data = read_reg(ser, mid, 56, 2)
                if data and len(data) == 2:
                    pos = data[0] | (data[1] << 8)
                    # Fetch model and firmware
                    model_data = read_reg(ser, mid, 3, 2)
                    model = (model_data[0] | (model_data[1] << 8)) if (model_data and len(model_data) == 2) else -1
                    fw_data = read_reg(ser, mid, 5, 1)
                    fw = fw_data[0] if (fw_data and len(fw_data) == 1) else -1
                    
                    print(f"      Motor ID {mid} | position {pos} | baud {baud} | Model: {model} | FW: {fw}")
                    found.append((mid, baud, pos, model, fw))
        else:
            hits = []
            for mid in range(1, 254):
                if try_ping(ser, mid):
                    data = read_reg(ser, mid, 56, 2)
                    pos  = (data[0] | (data[1] << 8)) if (data and len(data) == 2) else -1
                    model_data = read_reg(ser, mid, 3, 2)
                    model = (model_data[0] | (model_data[1] << 8)) if (model_data and len(model_data) == 2) else -1
                    fw_data = read_reg(ser, mid, 5, 1)
                    fw = fw_data[0] if (fw_data and len(fw_data) == 1) else -1

                    hits.append((mid, baud, pos, model, fw))
                    found.append((mid, baud, pos, model, fw))
            if hits:
                print()
                for h in hits:
                    print(f"      Motor ID {h[0]} | position {h[2]} | baud {h[1]} | Model: {h[3]} | FW: {h[4]}")
            else:
                print("no response")

        ser.close()

    return found

# -- Stage 3: Diagnosis ------------------------------------------------------
def diagnose(found):
    print("\n==========================================")
    print(" STAGE 3 - Diagnosis")
    print("==========================================")

    if found:
        print(f"\n  [SUCCESS] Found {len(found)} servo(s):")
        for mid, baud, pos, model, fw in found:
            print(f"    Motor ID {mid}  |  baud {baud:,}  |  position {pos}  |  Model {model}  |  FW {fw}")
        print()
        if any(b != 1_000_000 for _, b, _, _, _ in found):
            print("  !  Servo is at a non-default baud rate.")
            print("     Update jaw_config.json / jaw_server.py BAUD accordingly.")
        if any(mid != 1 for mid, _, _, _, _ in found):
            print("  !  Motor ID is not 1.  Update jaw_config.json -> motor_id.")
    else:
        print("""
  No servo responded across all baud rates (1M-9600) and all IDs (1-253).
  Ranked causes (most likely first):

  1. POWER - Servo logic / motor power rail is OFF or under-voltage.
             Feetech SCS/SMS series needs 6-8.4 V (or 9-12 V for STS/SCS high-voltage).
             Measure the servo power pins with a multimeter.

  2. DATA WIRE OPEN - The single half-duplex DATA line between the CH343
             TX/RX junction and the servo JST connector is broken, loose,
             or unplugged.  Re-seat the JST and probe continuity.

  3. CH343 TX/RX NOT JOINED - The half-duplex wiring requires TX and RX
             to be connected together (usually via a 1 kOhm TX resistor),
             then to the servo DATA pin.  If they are wired separately,
             the servo hears nothing.

  4. SERVO EEPROM LOCK - Writing to register 33 (MODE) while torque was
             enabled can corrupt the EEPROM on some Feetech servos.
             Fix: hold the servo's RESET button (if present) while powering on,
             or use the Feetech Servo Debug Tool (Windows GUI) to restore defaults.

  5. SERVO DEAD - Rare.  If all of the above check out, the servo MCU may
             have failed.  Try a known-good servo on the same cable.
""")

def main():
    print("==========================================")
    print("  FEETECH SERVO FULL DIAGNOSTIC SCANNER  ")
    print("==========================================")

    ports = [p.device for p in serial.tools.list_ports.comports()]
    if not ports:
        print("[ERROR] No COM ports found!  Plug in the USB adapter.")
        return

    port = "COM6" if "COM6" in ports else ports[0]
    print(f"  Detected ports : {ports}")
    print(f"  Using port     : {port}")

    loopback_test(port)
    found = full_sweep(port)
    diagnose(found)

if __name__ == "__main__":
    main()
