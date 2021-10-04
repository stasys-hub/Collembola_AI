/* ARDUINO SKETCH
******************
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        hoverMacroCam
.py
Purpose:             An Arduino Sketch that control to control the hoverMacroCam system (3rd axis possible but not implemented here). This version is suitable for a Pentax full frame DSLR (tested on a k1 II).
******************/

#include "IRremote.h"
 
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
  IrSender.begin(12, false);
}

void switchOn() {
  switchTime = millis();
  if (switchTime - prevSwitchTime > debouncingDelay) {
    prevSwitchTime = switchTime;
    systemState = !systemState;
  }
}

void shotNow()
{
  const uint16_t irSignal[]{ 13000, 3000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
  1000, 1000, 1000, 1000, 1000, 1000, 1000 };
  IrSender.sendRaw(irSignal, sizeof(irSignal) / sizeof(irSignal[0]), 38);
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
  for (byte i = 0; i < 7; i++){
    if (!systemState) {return;}
    course(YstepPin, 1500, 385);
    delay(0);
    shotNow();
    delay(4500);
    for (byte j = 0; j < 4; j++){
      if (!systemState) {return;}  
      course(XstepPin, 1500, 480);
      delay(0);
      shotNow();
      delay(4500);
      }
    Xdir = !Xdir;
    digitalWrite(XdirPin, Xdir);
    }
  course(XstepPin, 2000, 1920);
  digitalWrite(YdirPin, !Ydir);
  course(YstepPin, 2000, 2695);
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
