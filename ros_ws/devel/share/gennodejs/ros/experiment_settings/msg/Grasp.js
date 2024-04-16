// Auto-generated. Do not edit!

// (in-package experiment_settings.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let GraspEvoContacts = require('./GraspEvoContacts.js');
let GraspEvoPose = require('./GraspEvoPose.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Grasp {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.bestGrasp = null;
      this.ranking = null;
      this.bestPose = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('bestGrasp')) {
        this.bestGrasp = initObj.bestGrasp
      }
      else {
        this.bestGrasp = new GraspEvoContacts();
      }
      if (initObj.hasOwnProperty('ranking')) {
        this.ranking = initObj.ranking
      }
      else {
        this.ranking = 0.0;
      }
      if (initObj.hasOwnProperty('bestPose')) {
        this.bestPose = initObj.bestPose
      }
      else {
        this.bestPose = new GraspEvoPose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Grasp
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [bestGrasp]
    bufferOffset = GraspEvoContacts.serialize(obj.bestGrasp, buffer, bufferOffset);
    // Serialize message field [ranking]
    bufferOffset = _serializer.float32(obj.ranking, buffer, bufferOffset);
    // Serialize message field [bestPose]
    bufferOffset = GraspEvoPose.serialize(obj.bestPose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Grasp
    let len;
    let data = new Grasp(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [bestGrasp]
    data.bestGrasp = GraspEvoContacts.deserialize(buffer, bufferOffset);
    // Deserialize message field [ranking]
    data.ranking = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [bestPose]
    data.bestPose = GraspEvoPose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += GraspEvoContacts.getMessageSize(object.bestGrasp);
    length += GraspEvoPose.getMessageSize(object.bestPose);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'experiment_settings/Grasp';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '07436bacd6c0331b872013638b94fe36';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    GraspEvoContacts bestGrasp
    float32 ranking
    GraspEvoPose bestPose
    
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
    MSG: experiment_settings/GraspEvoContacts
    Header header
    sensor_msgs/PointCloud2 graspContactPoints
    
    ================================================================================
    MSG: sensor_msgs/PointCloud2
    # This message holds a collection of N-dimensional points, which may
    # contain additional information such as normals, intensity, etc. The
    # point data is stored as a binary blob, its layout described by the
    # contents of the "fields" array.
    
    # The point cloud data may be organized 2d (image-like) or 1d
    # (unordered). Point clouds organized as 2d images may be produced by
    # camera depth sensors such as stereo or time-of-flight.
    
    # Time of sensor data acquisition, and the coordinate frame ID (for 3d
    # points).
    Header header
    
    # 2D structure of the point cloud. If the cloud is unordered, height is
    # 1 and width is the length of the point cloud.
    uint32 height
    uint32 width
    
    # Describes the channels and their layout in the binary data blob.
    PointField[] fields
    
    bool    is_bigendian # Is this data bigendian?
    uint32  point_step   # Length of a point in bytes
    uint32  row_step     # Length of a row in bytes
    uint8[] data         # Actual point data, size is (row_step*height)
    
    bool is_dense        # True if there are no invalid points
    
    ================================================================================
    MSG: sensor_msgs/PointField
    # This message holds the description of one point entry in the
    # PointCloud2 message format.
    uint8 INT8    = 1
    uint8 UINT8   = 2
    uint8 INT16   = 3
    uint8 UINT16  = 4
    uint8 INT32   = 5
    uint8 UINT32  = 6
    uint8 FLOAT32 = 7
    uint8 FLOAT64 = 8
    
    string name      # Name of field
    uint32 offset    # Offset from start of point struct
    uint8  datatype  # Datatype enumeration, see above
    uint32 count     # How many elements in the field
    
    ================================================================================
    MSG: experiment_settings/GraspEvoPose
    Header header
    geometry_msgs/Vector3[] graspPosePoints
    geometry_msgs/Pose midPointPose
    
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
    const resolved = new Grasp(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.bestGrasp !== undefined) {
      resolved.bestGrasp = GraspEvoContacts.Resolve(msg.bestGrasp)
    }
    else {
      resolved.bestGrasp = new GraspEvoContacts()
    }

    if (msg.ranking !== undefined) {
      resolved.ranking = msg.ranking;
    }
    else {
      resolved.ranking = 0.0
    }

    if (msg.bestPose !== undefined) {
      resolved.bestPose = GraspEvoPose.Resolve(msg.bestPose)
    }
    else {
      resolved.bestPose = new GraspEvoPose()
    }

    return resolved;
    }
};

module.exports = Grasp;
