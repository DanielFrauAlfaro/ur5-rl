// Auto-generated. Do not edit!

// (in-package experiment_settings.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class GraspEvoPose {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.graspPosePoints = null;
      this.midPointPose = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('graspPosePoints')) {
        this.graspPosePoints = initObj.graspPosePoints
      }
      else {
        this.graspPosePoints = [];
      }
      if (initObj.hasOwnProperty('midPointPose')) {
        this.midPointPose = initObj.midPointPose
      }
      else {
        this.midPointPose = new geometry_msgs.msg.Pose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GraspEvoPose
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [graspPosePoints]
    // Serialize the length for message field [graspPosePoints]
    bufferOffset = _serializer.uint32(obj.graspPosePoints.length, buffer, bufferOffset);
    obj.graspPosePoints.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [midPointPose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.midPointPose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GraspEvoPose
    let len;
    let data = new GraspEvoPose(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [graspPosePoints]
    // Deserialize array length for message field [graspPosePoints]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.graspPosePoints = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.graspPosePoints[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [midPointPose]
    data.midPointPose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 24 * object.graspPosePoints.length;
    return length + 60;
  }

  static datatype() {
    // Returns string type for a message object
    return 'experiment_settings/GraspEvoPose';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'edfddb404e31fac0fe15cbe5e0286026';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    geometry_msgs/Vector3[] graspPosePoints
    geometry_msgs/Pose midPointPose
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GraspEvoPose(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.graspPosePoints !== undefined) {
      resolved.graspPosePoints = new Array(msg.graspPosePoints.length);
      for (let i = 0; i < resolved.graspPosePoints.length; ++i) {
        resolved.graspPosePoints[i] = geometry_msgs.msg.Vector3.Resolve(msg.graspPosePoints[i]);
      }
    }
    else {
      resolved.graspPosePoints = []
    }

    if (msg.midPointPose !== undefined) {
      resolved.midPointPose = geometry_msgs.msg.Pose.Resolve(msg.midPointPose)
    }
    else {
      resolved.midPointPose = new geometry_msgs.msg.Pose()
    }

    return resolved;
    }
};

module.exports = GraspEvoPose;
