#include <SPI.h>
#include <DW1000.h>

// connection pins
const uint8_t PIN_RST = 9; // reset pin
const uint8_t PIN_IRQ = 2; // irq pin
const uint8_t PIN_SS = SS; // spi select pin

// DEBUG packet sent status and count **** FROM RECEIVER ****
volatile boolean received = false;
volatile boolean error = false;
volatile int16_t numReceived = 0; // todo check int type
String message;

// DEBUG packet sent status and count **** FROM TRANSMITTER ****
boolean sent = false;
volatile boolean sentAck = false;
volatile unsigned long delaySent = 0;
int16_t sentNum = 0; // todo check int type
DW1000Time sentTime;


void setup() 
{
  //Identical for both
  Serial.begin(9600);
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
  String msg = "Hello DW1000, it's #"; msg += sentNum;
  DW1000.setData(msg);
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
  // continue on success confirmation
  // (we are here after the given amount of send delay time has passed)
  sentAck = false;
  // update and print some information about the sent message
  Serial.print("ARDUINO delay sent [ms] ... "); Serial.println(millis() - delaySent);
  DW1000Time newSentTime;
  DW1000.getTransmitTimestamp(newSentTime);
  Serial.print("Processed packet ... #"); Serial.println(sentNum);
  Serial.print("Sent timestamp ... "); Serial.println(newSentTime.getAsMicroSeconds());
  // note: delta is just for simple demo as not correct on system time counter wrap-around
  Serial.print("DW1000 delta send time [ms] ... "); Serial.println((newSentTime.getAsMicroSeconds() - sentTime.getAsMicroSeconds()) * 1.0e-3);
  sentTime = newSentTime;
  sentNum++;
  // again, transmit some data
  transmitter();


  //******* RECEIVER LOOP ***********
  // enter on confirmation of ISR status change (successfully received)
  if (received) {
    numReceived++;
    // get data as string
    DW1000.getData(message);
    Serial.print("Received message ... #"); Serial.println(numReceived);
    Serial.print("Data is ... "); Serial.println(message);
    Serial.print("FP power is [dBm] ... "); Serial.println(DW1000.getFirstPathPower());
    Serial.print("RX power is [dBm] ... "); Serial.println(DW1000.getReceivePower());
    Serial.print("Signal quality is ... "); Serial.println(DW1000.getReceiveQuality());
    received = false;
  }
  if (error) {
    Serial.println("Error receiving a message");
    error = false;
    DW1000.getData(message);
    Serial.print("Error data is ... "); Serial.println(message);
  }
  
}
