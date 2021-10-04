/* ARDUINO SKETCH
******************
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        hoverMacroCam
.py
Purpose:             An Arduino Sketch that control to control the hoverMacroCam system (3rd axis possible but not implemented here). This version is suitable for a Canon aps-c DSLR (tested on EOS 7D).
******************/

volatile bool systemState = LOW;
volatile long switchTime = 0;
volatile long prevSwitchTime = 0;
const byte interruptPin = 2;
const byte XdirPin = 3;
const byte XstepPin = 4;
const byte YdirPin = 5;
const byte YstepPin = 6;
volatile byte systemLedPin = 13;
const int debouncingDelay = 400;

void setup() {
  // Set Pins 3--13 as output (yet 7 to 11 are not used):
  for (byte i = 3; i < 14; i++) {
    pinMode(i, OUTPUT);
  }
  pinMode(interruptPin, INPUT);
  digitalWrite(XdirPin, HIGH);
  digitalWrite(YdirPin, HIGH);
  digitalWrite(systemLedPin, systemState);
  attachInterrupt(digitalPinToInterrupt(interruptPin), switchOn, RISING);
}

void switchOn() {
  switchTime = millis();
  if (switchTime - prevSwitchTime > debouncingDelay) {
    prevSwitchTime = switchTime;
    systemState = !systemState;
  }
}

void wait(unsigned int time){
  delayMicroseconds(time);
}

void high(int pinLED, int freq, int time){
  int pause = (1000/freq/2)-4;
  
  for (byte i = 0; i < time; i++) {
  digitalWrite(pinLED,HIGH);
  delayMicroseconds(pause);
  digitalWrite(pinLED,LOW);
  delayMicroseconds(pause);
  }
}

void shotNow()
{
  high(12,33,16);
  wait(7330);
  high(12,33,16);
}

void course(byte stepPin, int pulseFreq, int numSteps) {
    for (int i = 0; i < numSteps; i++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(pulseFreq);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(pulseFreq);
      }
}
void scanStage() {
  bool Xdir = LOW;
  bool Ydir = LOW;
  digitalWrite(YdirPin, Ydir); /* LOW = Y forward */
  digitalWrite(XdirPin, Xdir); /* LOW = X left */
  for (byte i = 0; i < 10; i++){
    if (!systemState) {return;}
    course(YstepPin, 500., 300);
    delay(1000);
    shotNow();
    delay(300);
    for (byte j = 0; j < 7; j++){
      if (!systemState) {return;}  
      course(XstepPin, 5000, 400);
      delay(1000);
      shotNow();
      delay(300);
      }
    Xdir = !Xdir;
    digitalWrite(XdirPin, Xdir);
    }
  digitalWrite(YdirPin, !Ydir);
  course(YstepPin, 5000, 5700);
}

void loop() {
  // Run one when button is pushed then wait for next push
  if (systemState){
    digitalWrite(systemLedPin, systemState);
    scanStage();
    systemState = LOW;
    digitalWrite(systemLedPin, systemState);
  }
}
