# This represents a line in 3d space.
# The slope is a normalized vector representing the direction of the line.
# The intercept is a point on the line.

# Example use case: given Z position, find X and Y position on the line.
# t = (z - intercept.z) / slope.z
# x = interecept.x + slope.x * t
# y = intercept.y + slope.y * t

geometry_msgs/Vector3 slope
geometry_msgs/Vector3 intercept