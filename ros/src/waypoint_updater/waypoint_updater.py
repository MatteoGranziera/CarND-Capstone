#!/usr/bin/env python

from os import close
import rospy
from geometry_msgs.msg import PoseStamped
from rospy.timer import sleep
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # TODO: Add a subscriber for /obstacle_waypoint below
        # rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_waypoint_index = -1

        # rospy.spin()
        self.loop()

    def loop(self):
        # Set 50 Hertz updates
        rate = rospy.Rate(20)
        rospy.loginfo("Loop started!")
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                closest_waypoint_index = self.get_closest_waypoint_index()
                self.publish_waypoints(closest_waypoint_index)
            rate.sleep()
    
    def get_closest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_index = self.waypoints_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_index]
        prev_coord = self.waypoints_2d[closest_index-1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_index = (closest_index + 1) % len(self.waypoints_2d)
        return closest_index

    def publish_waypoints(self, closest_index):
        lane = Lane()
        
        farthest_index = closest_index + LOOKAHEAD_WPS
        waypoints = self.base_waypoints.waypoints[closest_index:farthest_index]

        if self.stopline_waypoint_index == -1 or self.stopline_waypoint_index >= farthest_index:
            lane.waypoints = waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(waypoints, closest_index)
        
        self.final_waypoints_pub.publish(lane)

    def decelerate_waypoints(self, waypoints, closest_index):
        final_waypoints = []

        for i, wp in enumerate(waypoints):
            waypoint = Waypoint()
            waypoint.pose = waypoint.pose

            stop_index = max(self.stopline_waypoint_index - closest_index - 3, 0)
            dist = self.distance(waypoints, i, stop_index)
            velocity = math.sqrt(2*MAX_DECEL*dist)

            if velocity < 1.0:
                velocity = 0

            waypoint.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            final_waypoints.append(waypoint)

        return final_waypoints

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):

        # This is called one time
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_waypoint_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
