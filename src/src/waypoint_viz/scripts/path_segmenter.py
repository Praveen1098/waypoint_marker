#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class PathSegmenter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('path_segmenter', anonymous=True)

        # Subscribers
        self.path_sub = rospy.Subscriber('/move_base/PathPlanner/plan', Path, self.path_callback)
        self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.costmap_callback)

        # Publishers
        self.valid_segments_pub = rospy.Publisher('valid_segments', Path, queue_size=10)
        self.marker_pub = rospy.Publisher('waypoint_markers', MarkerArray, queue_size=10)

        # Costmap and global path
        self.local_costmap = None
        self.global_path = None
        self.costmap_array = None
        self.resolution = None
        self.origin = None

    def path_callback(self, msg):
        self.global_path = msg.poses
        if self.local_costmap is not None:
            self.process_path()

    def costmap_callback(self, msg):
        # Convert the costmap message to a usable format
        self.local_costmap = msg
        self.costmap_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin.position
        if self.global_path is not None:
            self.process_path()

    def process_path(self):
        if not self.global_path or len(self.global_path) < 2:
            rospy.logwarn("Global path is empty or has less than two points. Cannot process path.")
            return

        # Discretize the path into segments and check their validity
        valid_segments = Path()
        valid_segments.header.frame_id = "map"
        valid_segments.header.stamp = rospy.Time.now()

        waypoints = []  # List to collect waypoints

        for i in range(len(self.global_path) - 1):
            p1 = self.global_path[i].pose.position
            p2 = self.global_path[i + 1].pose.position

            # Discretize segment
            segment_points = self.discretize_segment(p1, p2, num_points=10)

            # Check if segment is valid
            if self.is_segment_valid(segment_points):
                for point in segment_points:
                    new_pose = PoseStamped()
                    new_pose.header.frame_id = "map"
                    new_pose.pose.position = point
                    valid_segments.poses.append(new_pose)

                # Collect the last point of the valid segment as a waypoint
                waypoints.append(segment_points[-1])  # Append the last point of the segment

        # Publish the valid segments
        self.valid_segments_pub.publish(valid_segments)
        rospy.loginfo(f"Published {len(valid_segments.poses)} valid segments")

        # Create and publish markers for waypoints
        self.publish_waypoint_markers(waypoints)


    def discretize_segment(self, p1, p2, num_points):
        # Generate points between p1 and p2
        segment_points = []
        for t in np.linspace(0, 1, num_points):
            x = (1 - t) * p1.x + t * p2.x
            y = (1 - t) * p1.y + t * p2.y
            point = PoseStamped().pose.position
            point.x = x
            point.y = y
            point.z = 0.0  # Assuming a 2D plane
            segment_points.append(point)
        return segment_points

    def is_segment_valid(self, segment_points):
        # Check if each point in the segment is valid according to the local costmap
        for point in segment_points:
            if not self.is_point_free(point):
                return False
        return True

    def is_point_free(self, point):
        # Convert the point to costmap coordinates
        costmap_x = int((point.x - self.origin.x) / self.resolution)
        costmap_y = int((point.y - self.origin.y) / self.resolution)

        # Check if the coordinates are within the bounds of the costmap
        if 0 <= costmap_x < self.costmap_array.shape[1] and 0 <= costmap_y < self.costmap_array.shape[0]:
            cell_value = self.costmap_array[costmap_y, costmap_x]
            # Check if the cell is free (0 means free, 100 means occupied)
            return cell_value == 0
        return False  # Out of bounds means not free

    def publish_waypoint_markers(self, waypoints):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = waypoint
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # Size of the sphere
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # Color of the sphere
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha (transparency)

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        rospy.loginfo(f"Published {len(waypoints)} waypoint markers")

if __name__ == '__main__':
    try:
        path_segmenter = PathSegmenter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

