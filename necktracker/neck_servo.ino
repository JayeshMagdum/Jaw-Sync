#include <Servo.h>

Servo neckServo;
const int servoPin = 9;
int currentAngle = 90;

void setup() {
  Serial.begin(9600);
  neckServo.attach(servoPin);
  neckServo.write(currentAngle);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int angle = input.toInt();
    if (angle >= 0 && angle <= 180) {
      currentAngle = angle;
      neckServo.write(currentAngle);
    }
  }
}
