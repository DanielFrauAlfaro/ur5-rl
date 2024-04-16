// Auto-generated. Do not edit!

// (in-package canbus_interface.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class CanFrameSrvRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.arbitration_id = null;
      this.data = null;
    }
    else {
      if (initObj.hasOwnProperty('arbitration_id')) {
        this.arbitration_id = initObj.arbitration_id
      }
      else {
        this.arbitration_id = 0;
      }
      if (initObj.hasOwnProperty('data')) {
        this.data = initObj.data
      }
      else {
        this.data = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CanFrameSrvRequest
    // Serialize message field [arbitration_id]
    bufferOffset = _serializer.uint32(obj.arbitration_id, buffer, bufferOffset);
    // Serialize message field [data]
    bufferOffset = _arraySerializer.uint8(obj.data, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CanFrameSrvRequest
    let len;
    let data = new CanFrameSrvRequest(null);
    // Deserialize message field [arbitration_id]
    data.arbitration_id = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [data]
    data.data = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.data.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'canbus_interface/CanFrameSrvRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ba91a1cf16899463e5da6dd76315ff65';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint32 arbitration_id
    uint8[] data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CanFrameSrvRequest(null);
    if (msg.arbitration_id !== undefined) {
      resolved.arbitration_id = msg.arbitration_id;
    }
    else {
      resolved.arbitration_id = 0
    }

    if (msg.data !== undefined) {
      resolved.data = msg.data;
    }
    else {
      resolved.data = []
    }

    return resolved;
    }
};

class CanFrameSrvResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CanFrameSrvResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CanFrameSrvResponse
    let len;
    let data = new CanFrameSrvResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'canbus_interface/CanFrameSrvResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CanFrameSrvResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: CanFrameSrvRequest,
  Response: CanFrameSrvResponse,
  md5sum() { return '675dd66a5938847259a403df984d1151'; },
  datatype() { return 'canbus_interface/CanFrameSrv'; }
};
