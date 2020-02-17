#include <SPI.h>
#include <DW1000.h>
#include "DW1000Ranging.h"

// connection pins
const uint8_t PIN_RST = 9; // reset pin
const uint8_t PIN_IRQ = 2; // irq pin
const uint8_t PIN_SS = SS; // spi select pin

// DEBUG packet sent status and count **** FROM RECEIVER ****
volatile boolean received = false;
volatile boolean error = false;
volatile int16_t numReceived = 0; // todo check int type
byte message[6];     //  [ ID_1, ID_2, FLOAT[0], FLOAT[1], FLOAT[2], FLOAT[3] ]

// DEBUG packet sent status and count **** FROM TRANSMITTER ****
boolean sent = false;
volatile boolean sentAck = false;
volatile unsigned long delaySent = 0;
int16_t sentNum = 0; // todo check int type
DW1000Time sentTime;


void setup() 
{
  Serial.begin(115200);
  delay(1000);

  //init the configuration
  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_SS);
  DW1000.newConfiguration();
  DW1000.setDefaults();

  DW1000.setDeviceAddress(1); // 5 = Sender, 6 = Receiver, 1 = pingPong
  DW1000.setNetworkId(10);
  DW1000.enableMode(DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
  DW1000.commitConfiguration();

  // attach callback for (successfully) received messages
  DW1000.attachReceivedHandler(handleReceived);
  DW1000.attachReceiveFailedHandler(handleError);
  DW1000.attachErrorHandler(handleError);
  receiver();

  // attach callback for (successfully) sent messages
  DW1000.attachSentHandler(handleSent);
  transmitter();

  //Range
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ); //Reset, CS, IRQ pin
  //define the sketch as anchor. It will be great to dynamically change the type of module
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachBlinkDevice(newBlink);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C", DW1000.MODE_LONGDATA_RANGE_ACCURACY);
}

// *********** RANGING FUNCTIONS ****************
void newRange() {
  Serial.print("from: "); Serial.print(DW1000Ranging.getDistantDevice()->getShortAddress(), HEX);
  Serial.print("\t Range: "); Serial.print(DW1000Ranging.getDistantDevice()->getRange()); Serial.print(" m");
  Serial.print("\t RX power: "); Serial.print(DW1000Ranging.getDistantDevice()->getRXPower()); Serial.println(" dBm");
}

void newBlink(DW1000Device* device) {
  Serial.print("blink; 1 device added ! -> ");
  Serial.print(" short:");
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device* device) {
  Serial.print("delete inactive device: ");
  Serial.println(device->getShortAddress(), HEX);
}

// **********  RECEIVER FUCNTIONS ***************
void handleReceived() {
  // status change on reception success
  received = true;
}
void handleError() {
  error = true;
}
void receiver() {
  DW1000.newReceive();
  DW1000.setDefaults();
  // so we don't need to restart the receiver manually
  DW1000.receivePermanently(true);
  DW1000.startReceive();
}


// **********  TRANSMITTER FUCNTIONS ***************
void handleSent() {
  // status change on sent success
  sentAck = true;
}

void transmitter() {
  // transmit some data
  Serial.print("Transmitting packet ... #"); Serial.println(sentNum);
  DW1000.newTransmit();
  DW1000.setDefaults();
  byte msg[6] = { 0, 2, 0, sentNum, 0, 0 }; //"Hello DW1000, it's #"; msg += sentNum;
  DW1000.setData(msg, 6);
  // delay sending the message for the given amount
  DW1000Time deltaTime = DW1000Time(10, DW1000Time::MILLISECONDS);
  DW1000.setDelay(deltaTime);
  DW1000.startTransmit();
  delaySent = millis();
}

void loop() 
{
  //******* TRANSMITTER LOOP *********
  if (!sentAck) {
    return;
  }
  sentAck = false;
  sentNum++;
  transmitter();

  //******* RECEIVER LOOP ***********
  // enter on confirmation of ISR status change (successfully received)
  if (received) {
    numReceived++;
    // get data as string
    DW1000.getData(message, 6);
    Serial.print("Received message ... #"); Serial.println(numReceived);
    Serial.print("Data is ... "); //Serial.println(message);
    for(int i = 0; i < 5; i++)
    {
      Serial.print(message[i]);
    }
    Serial.println(message[5]);
    received = false;
  }
  if (error) {
    Serial.println("Error receiving a message");
    error = false;
    DW1000.getData(message);
  }
  DW1000Ranging.loop();
}
