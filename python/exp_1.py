from robolink import *   # RoboDK API
from robodk import *     # basic matrix operations

# Connect to the open RoboDK instance
RDK = Robolink()

# Get the robot by its name in the station tree
robot = RDK.Item('Motoman HP6', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("Robot 'Motoman HP6' not found. Check the name in RoboDK tree.")

# Optional: print current joints
print("Current joints:", robot.Joints().list())

# Define a simple home position (in degrees)
home_joints = [0, 0, 0, 0, 0, 0]

# Define another joint position to move to
pose1_joints = [0, -45, 45, 0, 45, 0]

# Move robot
robot.MoveJ(home_joints)
robot.MoveJ(pose1_joints)
robot.MoveJ(home_joints)

print("Done.")