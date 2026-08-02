# ESP32 eyes serial patch

Your PC cannot open `http://192.168.4.1` because it has no Wi-Fi interface connected to the ESP32 AP. Use USB serial instead.

Keep your existing sketch. Add this parser code to it.

## 1. Add near the other global variables

```cpp
String serialLine = "";

float normToAngle(float v) {
  v = constrain(v, 0.0, 1.0);
  return 90.0 + v * 90.0;  // dashboard 0..1 maps to servo 90..180
}

String getTokenValue(String line, String key, String fallback = "") {
  int p = line.indexOf(key + "=");
  if (p < 0) return fallback;
  p += key.length() + 1;
  int e = line.indexOf(' ', p);
  if (e < 0) e = line.length();
  return line.substring(p, e);
}

void setServoTarget(int servo, float angle, int speed) {
  angle = constrain(angle, 90, 180);
  speed = constrain(speed, 1, 100);
  if (servo == 1) {
    loopMode1 = false;
    startAngle1 = currentAngle1;
    targetAngle1 = angle;
    moveSpeed1 = speed;
    Serial.printf("SERIAL S1 Target: %.0f | Speed: %d%%\n", targetAngle1, moveSpeed1);
  } else if (servo == 2) {
    loopMode2 = false;
    startAngle2 = currentAngle2;
    targetAngle2 = angle;
    moveSpeed2 = speed;
    Serial.printf("SERIAL S2 Target: %.0f | Speed: %d%%\n", targetAngle2, moveSpeed2);
  }
}

void handleSerialCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("look")) {
    float g13 = getTokenValue(line, "g13", "0.5").toFloat();
    float g14 = getTokenValue(line, "g14", getTokenValue(line, "g12", "0.5")).toFloat();
    int speed = getTokenValue(line, "speed", "75").toInt();
    setServoTarget(1, normToAngle(g13), speed);  // GPIO13 left/right
    setServoTarget(2, normToAngle(g14), speed);  // GPIO14 up/down
  } else if (line.startsWith("blink")) {
    setServoTarget(2, 180, 100);
    delay(90);
    setServoTarget(2, 135, 100);
  } else if (line.startsWith("mode")) {
    String state = getTokenValue(line, "state", "idle");
    if (state == "speaking") {
      setServoTarget(1, 135, 80);
      setServoTarget(2, 138, 80);
    } else if (state == "listening") {
      setServoTarget(1, 135, 60);
      setServoTarget(2, 150, 60);
    } else {
      setServoTarget(1, 135, 60);
      setServoTarget(2, 135, 60);
    }
  }
}

void pollSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleSerialCommand(serialLine);
      serialLine = "";
    } else if (c != '\r') {
      serialLine += c;
      if (serialLine.length() > 120) serialLine = "";
    }
  }
}
```

## 2. Add this as the first line inside `loop()`

```cpp
pollSerialCommands();
```

So your `loop()` becomes:

```cpp
void loop() {
  pollSerialCommands();
  server.handleClient();
  smoothMoveServo(myServo1, currentAngle1, targetAngle1, startAngle1,
                  moveSpeed1, lastMoveTime1, loopMode1, loopMin1, loopMax1, loopForward1);
  smoothMoveServo(myServo2, currentAngle2, targetAngle2, startAngle2,
                  moveSpeed2, lastMoveTime2, loopMode2, loopMin2, loopMax2, loopForward2);
}
```

After uploading, the dashboard can control eyes over USB with:

```text
look g13=0.50 g14=0.50
mode state=speaking
mode state=listening
mode state=idle
blink
```
