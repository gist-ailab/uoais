std_msgs/Header header

# bounding boxes in pixels
sensor_msgs/RegionOfInterest[] bboxes

# visible masks as Image
sensor_msgs/Image[] visible_masks

# amodal masks as Image
sensor_msgs/Image[] amodal_masks

# occlusions of each instances (0: not-occluded, 1: occluded)
int32[] occlusions

# String class names for each instances (default: object)
string[] class_names

# String class IDs for each instances (default: 0)
int32[] class_ids
