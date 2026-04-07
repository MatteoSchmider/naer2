# entirely new approach:
# populations are sets!
# => patterns generate 1D indices!
# populations can have optional per-node position meta-data (e.g. from real world 3D scan) which is a tuple of floats
# patterns can generate populations from a node instance just like the * operator can
# => generate list of indices => maximum generated index sets the population size
# chaining of patterns can also happen with the source being taken as the previous one's target
# the result object of a pattern chain of size > 0 is a naer.network
# the layers can be accessed via the .layers attribute, the connections via .connections, which return a list of populations and connections respectively, in order of addition
# net = node1() @ pattern * edge1() + pattern * edge10() >> node2() @ pattern: edge2() >> node3() @ pattern: edge3() >> node4()
# net.layers = [node1, node2, node3, node4]
# net.connections = [(pattern, edge1), (pattern, edge2), (pattern, edge3)]
# net.layers[0] @ pattern: edge4() >> net.layers[0]


# how is metadata used?
# return value from a pattern is a list of indices
# => list of indices means list of ints or list of tuples of ints
# if tuples of ints, then the generated population is a tensor! => immediately generate metadata for population from that
# pattern(src_node_position: tuple(int), optional_dst_pop_metadata list(tuple(int))) => list(tuple(int))

# where optional_dst_pop_metadata is technically a n-d array, with the shape

# what does pattern need?
# if the source population is a 1D shape/shapeless, the index of the source node as int
# if the source population is a tensor shape, the index of the source node as tuple of ints
# if the source population is a tensor with metadata, the index of the source node as tuple of ints along with metadata? or just the source node metadata?
# imagine we have a population from a 3d scan. the nodes only really make sense with the real position vector now, not the index, right?
# ======> pattern only needs the source node metadata!
# =========> we can implement alternative metadata later
# => call this: position!

# make interface here clean for offline adjacemncy, then adapt for online adjacency without losing features (maybe make things optional?)

# pattern interface is kinda crucial here => steps:
# 1.) make interface, program conv2d example with it
# 2.) make full connection syntax using this interface but return just a connection object
# 3.) build out a network object and return that instead
# 4.) implement population generation
# 5.) only now do online patterns


def test_pattern(src_position: tl.tensor, dst_region: tl.tensor) -> tl.tensor:
    return None


# src_position is tensor of shape (1, 1)
# dst_region is not needed
def test_conv2d(src_position):
    return naer.box(
        [
            [src_position[0] - 1, src_position[0] + 1],
            [src_position[1] - 1, src_position[1] + 1],
        ]
    )


# src_position is tensor of shape (1, 1)
# dst_region is tensor of shape (1, 1) => interpretable as 2D image/plane
def test_conv2d_shape(src_position, dst_region: naer.shape):
    # test that the box is inside of shape
    return naer.box(
        [
            [src_position[0] - 1, src_position[0] + 1],
            [src_position[1] - 1, src_position[1] + 1],
        ]
    )


# src_position is tensor of shape (1, 1)
# dst_region is tensor of shape (2, 2) => interpretable as a rectangle inside a larger image
def test_conv2d_box(src_position, dst_region: naer.box):
    # test that the box is inside of the box
    return naer.box(
        [
            [src_position[0] - 1, src_position[0] + 1],
            [src_position[1] - 1, src_position[1] + 1],
        ]
    )


# src_position is tensor of shape (1, 1)
# dst_region is tensor of shape (500, 1, 1) => interpretable as a list of 500 possible 2D positions
def test_conv2d_points(src_position, dst_region: naer.points):
    return naer.points(
        [
            [src_position[0] - 1, src_position[1] - 1],
            [src_position[0], src_position[1] - 1],
            [src_position[0] + 1, src_position[1] - 1],
            [src_position[0] - 1, src_position[1]],
            [src_position[0], src_position[1]],
            [src_position[0] + 1, src_position[1]],
            [src_position[0] - 1, src_position[1] + 1],
            [src_position[0], src_position[1] + 1],
            [src_position[0] + 1, src_position[1] + 1],
        ]
    )


# => region is one of "shape", "box", "points" or a tensor with no guarantees or None/optional param
# => src index is "position"
# => return value is always one of "shape", "box", "points", or a tensor with no guarantees


import numpy as np


# A hypothetical API function
def generate_position(start_position: np.ndarray, region=None, callback=None):
    """
    Generates a new position based on a starting position and an optional region.

    Args:
        start_position (np.ndarray): The initial n-dimensional position.
        region (Optional[np.ndarray]): The region tensor (Shape, Points, or Box).
        callback (Optional[callable]): A user-defined function to generate the new position.

    Returns:
        np.ndarray: The new n-dimensional position.
    """
    if callback:
        # User provides a custom logic
        return callback(start_position, region)
    else:
        # Default logic (e.g., a small random walk if no region is given)
        # or specific logic based on the region type.
        if region is None:
            return start_position + np.random.uniform(-0.1, 0.1, start_position.shape)
        # You would implement default logic for Shape, Points, and Box here
        # based on what makes sense for your application.
        # e.g., for a Box, you might return a random point within the box.
        # ...


# --- Example 1: No region, default behavior (e.g., a random walk)
start_pos = np.array([10.0, 20.0])
new_pos_default = generate_position(start_pos)
print(f"Default new position: {new_pos_default}")

# --- Example 2: Using a Shape region with a custom callback
# A 'Shape' region could be a single point the user wants to jump to.
region_shape = np.array([30.0, 40.0])


def jump_to_shape(start_pos, region):
    # The callback simply returns the position of the Shape.
    return region


new_pos_jump = generate_position(start_pos, region=region_shape, callback=jump_to_shape)
print(f"New position after jumping to Shape: {new_pos_jump}")

# --- Example 3: Using a Box region with a custom callback
# A 'Box' is a range: [min_x, max_x], [min_y, max_y]
region_box = np.array([[0.0, 50.0], [0.0, 50.0]])


def random_in_box(start_pos, region):
    # Generate a random point within the given Box.
    min_vals = region[:, 0]
    max_vals = region[:, 1]
    return np.random.uniform(min_vals, max_vals)


new_pos_box = generate_position(start_pos, region=region_box, callback=random_in_box)
print(f"New position within the Box: {new_pos_box}")

# --- Example 4: Using a Points region with a custom callback
# 'Points' is a list of specific coordinates to choose from.
region_points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])


def choose_from_points(start_pos, region):
    # Choose one of the points from the list at random.
    return region[np.random.randint(len(region))]


new_pos_points = generate_position(
    start_pos, region=region_points, callback=choose_from_points
)
print(f"New position chosen from Points list: {new_pos_points}")
