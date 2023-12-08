#include <Servo.h>. 

#define RED_LIGHT_DELAY   3000
#define GREEN_LIGHT_DELAY 3000
#define BUZZER_DELAY      500
#define BUZZER_PIN        3

#define RED_PIN           2
#define GREEN_PIN         4

#define AIM_PIN           9
#define TRIGGER_PIN       11
#define HEAD_PIN          13

#define AIM_DELAY         1250
#define SPIN_DELAY        2000
// #define TRIGGER_DELAY     1000
#define TRIGGER_DELAY     900
#define HEAD_DELAY        1000

// #define TRIGGER_NEU       120
// #define TRIGGER_FWD       110

#define TRIGGER_NEU       125
#define TRIGGER_FWD       117

// #define GUN_NEU           15
#define GUN_NEU           165

#define HEAD_REV          0
#define HEAD_FWD          180

String anglesString;
Servo gun;
Servo trigger;
Servo head;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, HIGH);
  gun.attach(AIM_PIN);
  gun.write(GUN_NEU);
  trigger.attach(TRIGGER_PIN);
  trigger.write(TRIGGER_NEU);
  head.attach(HEAD_PIN);
  head.write(HEAD_REV);
}

int j = 0;
void loop() {
  // put your main code here, to run repeatedly:

  anglesString = "";
  // Serial.readString(); // Flush buffer
  while (!Serial.available()) { continue; } // Do nothing
  anglesString = Serial.readString();

  if (anglesString.equals("#")) {
    digitalWrite(RED_PIN, HIGH);
    digitalWrite(GREEN_PIN, LOW);
    head.write(HEAD_FWD);
    delay(HEAD_DELAY);
    return;
  }

  if (anglesString.equals("!")) {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, HIGH);
    head.write(HEAD_REV);
    delay(HEAD_DELAY);
    Serial.println("*");
    delay(5);
    Serial.println("*");
    delay(5);
    Serial.println("*");
    return;
  }
  
  int angleCount = 0;
  for (int i = 0; i < anglesString.length(); i++) {
    if (anglesString.charAt(i) == ',') {
      angleCount++;
    }
  }
  for (int i = 0; i < angleCount; i++) {
    int charIndex = anglesString.indexOf(',');
    String substring = anglesString.substring(0, charIndex);
    if (i != angleCount - 1) {
      anglesString = anglesString.substring(charIndex + 1);
    }
    int angle = substring.toInt();
    gun.write(angle);
    delay(AIM_DELAY);
    trigger.write(TRIGGER_FWD);
    delay(TRIGGER_DELAY);
    trigger.write(TRIGGER_NEU);
  }
  gun.write(GUN_NEU);

  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, HIGH);

  head.write(HEAD_REV);
  delay(HEAD_DELAY);

  Serial.println(anglesString);
  delay(5);
  Serial.println(anglesString);
  delay(5);
  Serial.println(anglesString);
}
