#!/usr/bin/env python

class MineSweeperPath:
    def __init__(self, grid, start_node, max_length_between_waypoints):
        """
        Args:
        - grid (list of list): each element has a 0 or 1, but this info is not used
        - start_node (tuple of indices i,j): i is the row index, j is the col index
        - max_length_between_waypoints (int):
        """
        self.grid = grid
        self.start_node = start_node
        assert self.start_node == (0,0)
        self.height = len(grid)
        self.width = len(self.grid[0])
        self.max_length_between_waypoints = max_length_between_waypoints


    def is_inside_grid(self, i, j):
        """Checks if this node given by (i,j) is inside the grid

        Args:
            i (int): the row index
            j (int): the col index
        """
        return (j >= 0) and (j < self.width) and (i >= 0) and (i < self.height)


    def get_path(self):
        path = [] # path of (i,j), i is row j is col
        directions = ["right", "down", "left", "down"]
        next_d = 0 # index of next moving

        def next_direction(current_d):
            next_direction_index = current_d + 1
            if next_direction_index == len(directions):
                next_direction_index = 0
            return next_direction_index

        # this is the current position
        i, j = 0, 0
        while self.is_inside_grid(i, j):
            path.append((i,j))
            
            # move right until the end
            if directions[next_d] == "right":
                j = min(self.width-1, j + self.max_length_between_waypoints)
                if j == self.width-1:
                    next_d = next_direction(next_d)

            # move down once
            elif directions[next_d] == "down":
                i += 1
                next_d = next_direction(next_d)

            # move left until the end
            elif directions[next_d] == "left":
                j = max(0, j - self.max_length_between_waypoints)
                if j == 0:
                    next_d = next_direction(next_d)
            
            else:
                raise Exception("direction {} not found".format(directions[next_d]))
            
            
        return path
    
#FOR TESTING
if __name__== "__main__":
    grid = [[0]*15]*15
    start_node = (0, 0)
    for row in grid:
        print(row)

    sp = MineSweeperPath(grid, start_node, max_length_between_waypoints=3)
    path = sp.get_path()
    print("Path: {}".format(path))