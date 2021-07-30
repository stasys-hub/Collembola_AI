
/*******************************************

Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        hoverMacroCam
.py
Purpose:             A Script that control to control the hoverMacroCam system (3rd axis possible but not implemented here).
Sources: the infrared trigger signal information was found thanks to the cameraIrControl Library by Sebastian Setz at https://github.com/dharmapurikar/Arduino/blob/master/libraries/multiCameraIrControl. The library itself was not used because it was found to produce unreliable code
Licence:             BSD

 * Name.......:  hoverMacroCam control script
 * Description:  A script that control to control the hoverMacroCam system (3rd axis possible but not implemented here).
 * Author.....:  Clément Schneider
 * Version....:  0.5
 * Date.......:  2021-06-21
 * Project....:  
 * Contact....:  clement.schneider{a}senckenberg.de
              We distribute it here with a tiny fix for user conveniency. 
 *
 ********************************************/
 
/* From the Arduino IDE top menu: "Sketch > Include Library > Add .ZIP Library",
   then select the multiCameraIrControl folder "under Collembola_AI/immaging/" */
#include <multiCameraIrControl.h>

// Select your DSLR brand by commenting/uncommenting. So far I tested the library only with a Canon EOS 7D and a Pentax K-1
// Please refer to multiCameraIrControl repository to check the available brands: https://github.com/dharmapurikar/Arduino/blob/master/libraries/multiCameraIrControl
Pentax Kam(12);
// Canon Kam(12);
// Nikon Kam(12);

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

/*
void wait(unsigned int time){
  unsigned long start = micros();
  while(micros()-start<=time){
  }
//  delayMicroseconds(time);
}

void high(int pinLED, int freq, int time){
  int pause = (1000/freq/2)-4;
  //int cycles = time/
  
    unsigned long start = micros();
    while(micros()-start<=time){
//  for (byte i = 0; i < cycles; i++) {
    digitalWrite(pinLED,HIGH);
    delayMicroseconds(pause);
    digitalWrite(pinLED,LOW);
    delayMicroseconds(pause);
  }
}

 Useless trash, cleaning once final check
void shotNow()
{
  high(12,33,16);
  wait(7330);
  high(12,33,16);
}

// For pentax uncomment
void shotNow()
{
  high(12,38,13000);
  wait(3000);
  for (int i=0;i<7;i++){
    high(12,38,1000);
    wait(1000);
  };
}*/

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
    course(YstepPin, 1500, 200);
    delay(1500);
    Kam.shotNow();
    delay(400);
    for (byte j = 0; j < 7; j++){
      if (!systemState) {return;}  
      course(XstepPin, 1500, 280);
      delay(1500);
      Kam.shotNow();
      delay(400);
      }
    Xdir = !Xdir;
    digitalWrite(XdirPin, Xdir);
    }
  digitalWrite(YdirPin, !Ydir);
  course(YstepPin, 1500, 2000);
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
