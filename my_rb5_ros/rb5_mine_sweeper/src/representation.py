#!/usr/bin/env python
"""
The class in charge of making the world model
"""
import math
NODE_OCCUPIED = 0
NODE_TRAVERSABLE = 1

class WorldModel:
    def __init__(self, cell_length_m, width_m, height_m, safety_nodes, rectangle_obstacle):
        """Makes a world representation as a rectangular grid        

        Args:
            cell_length_m (float): size of each cell in meters
            width_m (float): width of the world in meters
            height_m (float): height of the world in meters
            safety_nodes (int): number of nodes to use as safety from obstacles and walls            
            rectangle_obstacle (list): list of 4 tuples (xi, yi), each representing corner i of the rectangle
        """
        self.cell_length_m = float(cell_length_m)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.safety_nodes = safety_nodes
        assert isinstance(safety_nodes, int)
        
        self.rectangle_obstacle = rectangle_obstacle
        if rectangle_obstacle is not None:
            assert isinstance(rectangle_obstacle, list)
            for xy in rectangle_obstacle:
                assert len(xy) == 2

        # Start building the model
        self.create_grid()
        if rectangle_obstacle is not None:
            self.insert_obstacle()        
        
        # Apply safety
        for _ in range(self.safety_nodes):
            self.apply_safety()


    def create_grid(self):
        """Creates a 2D grid with num_rows and num_cols.
        If a cell is traversable the value is going to be NODE_TRAVERSABLE,
        if not, is going to be NODE_OCCUPIED.

        World frame is:
        Y
        |
        |
        |__________X
        Origin

        Graph frame is:
        Origin
        ___________X
        |
        |
        |
        |
        Y
        
        """
        self.num_rows = int(math.ceil(self.height_m / self.cell_length_m))
        self.num_cols = int(math.ceil(self.width_m / self.cell_length_m))
        print("Grid has {} rows and {} columns".format(self.num_rows, self.num_cols))
        self.grid = []
        for _ in range(self.num_rows):
            self.grid.append([NODE_TRAVERSABLE] * self.num_cols)    

    def is_inside_grid(self, i, j):
        """Checks if this node given by (i,j) is inside the grid

        Args:
            i (int): the row index
            j (int): the col index
        """
        return (j >= 0) and (j < self.num_cols) and (i >= 0) and (i < self.num_rows)


    def point_to_node(self, x, y):
        """Given point (x,y) in world frame, and in meters, it transforms it
        to the corresponding cell in the grid (i,j) (row and col)
        """
        index_col = int(x / self.cell_length_m)
        index_row = self.num_rows - 1 - int(y / self.cell_length_m)

        if self.is_inside_grid(index_row, index_col):
            return (index_row, index_col)
        else:
            print("The point x,y = {},{} mapped to the graph is out of bounds. It would me mapped to row,col = {},{}".format(
                x, y, index_col, index_row))
            return None

    def node_to_point(self, i, j):
        """Given a node, this will return the center of that node in world coordinates

        Args:
        - i (row index): 
        - j (col index):

        Returns:
        - x (in meters)
        - y (in meters)
        """
        assert self.is_inside_grid(i, j)
        x = self.cell_length_m * (0.5 + float(j))
        y = self.cell_length_m * (0.5 + float(self.num_rows - 1 - i))
        return x, y

    def insert_obstacle(self):
        """
        Inserts the obstacle in the grid and marks its positions with zeros (1 is traversable)
        """
        for x, y in self.rectangle_obstacle: # x,y is a corner of the obstacle
            i, j = self.point_to_node(x, y)
            self.grid[i][j] = NODE_OCCUPIED

    def apply_safety(self):
        """
        Applies a safety pad of 1 to the non-traversable nodes.
        At every horizontal and vertical node that is non traversable, I will pad by 1
        """
        
        # Get the nodes occupied first
        non_traversable_indices = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.grid[i][j] == NODE_OCCUPIED:
                    non_traversable_indices.append((i,j))
        
        # Now pad them in these directions
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        for (i,j) in non_traversable_indices:
            for di, dj in directions:
                i_ = i + di # neighbor in row
                j_ = j + dj # neighbor in col
                if self.is_inside_grid(i_, j_):
                    self.grid[i_][j_] = NODE_OCCUPIED


    def print_grid(self, robot_node = None, path_nodes = None, target_node = None):
        
        # Print the cols
        header = "   "
        for j in range(self.num_cols):
            header += " " + str(j).zfill(2) + " "
        print(header)

        # Store the value to print in the map
        grid_to_print = []
        for i in range(self.num_rows):
            grid_to_print.append(["   "] * self.num_cols)
        
        if path_nodes is not None:
            for index, (i,j) in enumerate(path_nodes):
                grid_to_print[i][j] = " " + str(index).zfill(2) + " "
        
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.grid[i][j] == NODE_OCCUPIED:
                    grid_to_print[i][j] = " X "
                elif robot_node is not None and (i,j) == robot_node:
                    grid_to_print[i][j] = "  R "
                elif target_node is not None and (i,j) == target_node:
                    grid_to_print[i][j] = " T "
        

        for i in range(self.num_rows):
            row_str = str(i).zfill(2) + "|"
            for j in range(self.num_cols):
                row_str = row_str + grid_to_print[i][j] + "|"                
            print(row_str)

if __name__ == "__main__":
    model = WorldModel(
        cell_length_m=0.25,
        width_m=2,
        height_m=1,
        safety_nodes=1,
        robot_x=0,
        robot_y=0,
        rectangle_obstacle=[(0.75, 0.40), (0.75, 0.25), (1.24, 0.25), (1.24, 0.49)]
    )
    
    print("\nTesting point to node method")
    # Test 1 for point to node
    x,y = 0.3, 0.6
    n = model.point_to_node(x, y)
    assert (n[0] == 1 and n[1] == 1)
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    # Testing 2 for point to node
    x,y = 0.5, 0.75
    n = model.point_to_node(x, y)
    assert (n[0] == 0 and n[1] == 2)
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    # Testing 3 for point to node
    x,y = 0.49, 0.75
    n = model.point_to_node(x, y)
    assert (n[0] == 0 and n[1] == 1)
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    # Testing 4 for point to node
    x,y = 0, 0
    n = model.point_to_node(x, y)
    assert (n[0] == 3 and n[1] == 0)
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    # Testing 5 for point to node
    x,y = 0, 1
    n = model.point_to_node(x, y)
    assert n is None
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    # Testing 6 for point to node
    x,y = 0.75, 0.4
    n = model.point_to_node(x, y)
    assert (n[0] == 2 and n[1] == 3)    
    print("Point x,y = {},{} is row,col = {}".format(x,y,n))

    print("\nTesting node to point")
    i,j = 0, 0
    x,y = model.node_to_point(i,j)
    assert (x == 0.25/2 and y == 0.25/2 + 3*0.25)  
    print("Node i,j = {},{} is x,y = {},{}".format(i,j,x,y))
    i,j = 3, 7
    x,y = model.node_to_point(i,j)
    assert (x == 2 - 0.25/2 and y == 0.25/2)  
    print("Node i,j = {},{} is x,y = {},{}".format(i,j,x,y))