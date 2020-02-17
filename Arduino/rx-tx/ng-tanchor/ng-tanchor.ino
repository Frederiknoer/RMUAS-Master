#include <SPI.h>
#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>
#include <DW1000NgRanging.hpp>

// connection pins
const uint8_t PIN_RST = 9; // reset pin
const uint8_t PIN_IRQ = 2; // irq pin
const uint8_t PIN_SS = SS; // spi select pin

// messages used in the ranging protocol
// TODO replace by enum
#define POLL 0
#define POLL_ACK 1
#define RANGE 2
#define RANGE_REPORT 3
#define RANGE_FAILED 255

volatile byte expectedMsgId = POLL;

volatile boolean sentAck = false;
volatile boolean receivedAck = false;
// timestamps to remember
uint64_t timePollSent;
uint64_t timePollReceived;
uint64_t timePollAckSent;
uint64_t timePollAckReceived;
uint64_t timeRangeSent;
uint64_t timeRangeReceived;
int partnerID;

boolean protocolFailed = false;

uint64_t timeComputedRange;
// data buffer
#define LEN_DATA 17
byte data[LEN_DATA];
// watchdog and reset period
uint32_t lastActivity;
uint32_t resetPeriod = 250;
// reply times (same on both sides for symm. ranging)
uint16_t replyDelayTimeUS = 3000;
uint16_t successRangingCount = 0;
uint32_t rangingCountPeriod = 0;
float samplingRate = 0;

device_configuration_t DEFAULT_CONFIG = {
    false,
    true,
    true,
    true,
    false,
    SFDMode::STANDARD_SFD,
    Channel::CHANNEL_5,
    DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ,
    PreambleLength::LEN_256,
    PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
    true,
    true,
    true,
    false,
    true
};


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
  DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
  DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);

  DW1000Ng::setNetworkId(10);
  DW1000Ng::setDeviceAddress(1);
  DW1000Ng::setAntennaDelay(16436);

  char msg[128];
  DW1000Ng::attachSentHandler(handleSent);
  DW1000Ng::attachReceivedHandler(handleReceived);

  transmitPoll();
  receiver();
  noteActivity();
  // for first time ranging frequency computation
  rangingCountPeriod = millis();

}

void noteActivity() {
    // update activity timestamp, so that we do not reach "resetPeriod"
    lastActivity = millis();
}

void resetInactive() {
    // tag sends POLL and listens for POLL_ACK
    expectedMsgId = POLL_ACK;
    DW1000Ng::forceTRxOff();
    transmitPoll();
    noteActivity();
}

void handleSent() {
    // status change on sent success
    sentAck = true;
}

void handleReceived() {
    // status change on received success
    receivedAck = true;
}

void transmitPollAck() {
    data[0] = POLL_ACK;
    DW1000Ng::setTransmitData(data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void transmitRangeReport(float curRange, int recvID) {
    data[0] = RANGE_REPORT;
    // write final ranging result
    memcpy(data + 1, &curRange, 4);
    DW1000Ng::getDeviceAddress(data + 5);
    data[6] = recvID;
    DW1000Ng::setTransmitData(data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void transmitRangeFailed() {
    data[0] = RANGE_FAILED;
    DW1000Ng::setTransmitData(data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void receiver() {
    DW1000Ng::forceTRxOff();
    // so we don't need to restart the receiver manually
    DW1000Ng::startReceive();
}

void transmitPoll() {
    data[0] = POLL;
    DW1000Ng::setTransmitData(data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void transmitRange() {
  data[0] = RANGE;

  /* Calculation of future time */
  byte futureTimeBytes[LENGTH_TIMESTAMP];

  timeRangeSent = DW1000Ng::getSystemTimestamp();
  timeRangeSent += DW1000NgTime::microsecondsToUWBTime(replyDelayTimeUS);
  DW1000NgUtils::writeValueToBytes(futureTimeBytes, timeRangeSent, LENGTH_TIMESTAMP);
  DW1000Ng::setDelayedTRX(futureTimeBytes);
  timeRangeSent += DW1000Ng::getTxAntennaDelay();

  DW1000NgUtils::writeValueToBytes(data + 1, timePollSent, LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 6, timePollAckReceived, LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 11, timeRangeSent, LENGTH_TIMESTAMP);
  DW1000Ng::getDeviceAddress(data + 16);
  
  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit(TransmitMode::DELAYED);
  //Serial.print("Expect RANGE to be sent @ "); Serial.println(timeRangeSent.getAsFloat());
}

void loop() {
  int32_t curMillis = millis();
  if (!sentAck && !receivedAck) {
    // check if inactive
    if (millis() - lastActivity > resetPeriod) {
      resetInactive();
      }
    return;
    }

  if (sentAck) {
    sentAck = false;
    byte msgId = data[0];
    if (msgId == POLL_ACK) {
      timePollAckSent = DW1000Ng::getTransmitTimestamp();
      noteActivity();
      }
    DW1000Ng::startReceive();
    }

    if (receivedAck) {
        receivedAck = false;
        // get message and parse
        DW1000Ng::getReceivedData(data, LEN_DATA);
        byte msgId = data[0];
        Serial.print("Msg ID: "); Serial.println(msgId); 
        Serial.print("Msg expected: "); Serial.println(expectedMsgId);
        if (msgId != expectedMsgId) {
            // unexpected message, start over again
            //Serial.print("Received wrong message # "); Serial.println(msgId);
            expectedMsgId = POLL_ACK;
            transmitPoll();
            return;
        }
        if (msgId == POLL) {
            // on POLL we (re-)start, so no protocol failure
            protocolFailed = false;
            timePollReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = RANGE;
            transmitPollAck();
            noteActivity();
        } else if (msgId == POLL_ACK) {
            timePollSent = DW1000Ng::getTransmitTimestamp();
            timePollAckReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = RANGE_REPORT;
            transmitRange();
            noteActivity();
        } else if (msgId == RANGE_REPORT) {
            expectedMsgId = POLL_ACK;
            //float curRange;
            //memcpy(&curRange, data + 1, 4);
            
            double distance = DW1000NgUtils::bytesAsValue(data + 1, 4);
            String rangeString = "Range: "; rangeString += distance; rangeString += " m";
            rangeString += "\t Between ID: "; rangeString += data[5]; rangeString+= "&"; rangeString += data[6];
            
            transmitPoll();
            noteActivity();
        } else if (msgId == RANGE_FAILED) {
            expectedMsgId = POLL_ACK;
            transmitPoll();
            noteActivity();
        } else if (msgId == RANGE) {
            timeRangeReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = POLL;
            if (!protocolFailed) {
                timePollSent = DW1000NgUtils::bytesAsValue(data + 1, LENGTH_TIMESTAMP);
                timePollAckReceived = DW1000NgUtils::bytesAsValue(data + 6, LENGTH_TIMESTAMP);
                timeRangeSent = DW1000NgUtils::bytesAsValue(data + 11, LENGTH_TIMESTAMP);
                partnerID = data[16];
                // (re-)compute range as two-way ranging is done
                double distance = DW1000NgRanging::computeRangeAsymmetric(timePollSent,
                                                            timePollReceived, 
                                                            timePollAckSent, 
                                                            timePollAckReceived, 
                                                            timeRangeSent, 
                                                            timeRangeReceived);
                /* Apply simple bias correction */
                distance = DW1000NgRanging::correctRange(distance);
                
                String rangeString = "Range: "; rangeString += distance; rangeString += " m";
                rangeString += "\t RX power: "; rangeString += DW1000Ng::getReceivePower(); rangeString += " dBm";
                //rangeString += "\t Sampling: "; rangeString += samplingRate; rangeString += " Hz";
                rangeString += "\t From ID: "; rangeString += partnerID;
                Serial.println(rangeString);
                //Serial.print("FP power is [dBm]: "); Serial.print(DW1000Ng::getFirstPathPower());
                //Serial.print("RX power is [dBm]: "); Serial.println(DW1000Ng::getReceivePower());
                //Serial.print("Receive quality: "); Serial.println(DW1000Ng::getReceiveQuality());
                // update sampling rate (each second)
                transmitRangeReport(distance * DISTANCE_OF_RADIO_INV, partnerID);
                successRangingCount++;
                /*
                if (curMillis - rangingCountPeriod > 1000) {
                    samplingRate = (1000.0f * successRangingCount) / (curMillis - rangingCountPeriod);
                    rangingCountPeriod = curMillis;
                    successRangingCount = 0;
                }*/
            }
            else {
                transmitRangeFailed();
            }

            noteActivity();
        }

    }


}
