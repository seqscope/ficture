import sys, io, os
import numpy as np


class BS2DNode:
    """
    Node object for simple binary 2D partition
    """
    def __init__(self, level, coord_min, coord_max, left=None, right=None, midpt=None, split=None, data=None):
        self.level = level
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.left = left
        self.right = right
        self.midpt = midpt  # Split point
        self.split = split  # Split along which dimension
        self.data = data

    def printnode(self):
        print(f"Level {self.level}, coordinates: ({self.coord_min[0]:.2f}, {self.coord_min[1]:.2f}) - ({self.coord_max[0]:.2f}, {self.coord_max[1]:.2f})")
        if self.data is None:
            print(f"\tInternal node spliting along the {self.split}-th dim at {self.midpt:.2f}")
        else:
            print(f"\tLeaf node with {len(self.data)} points")


class BS2DTree:
    """
    KD tree for simple binary 2D partition
    """
    def __init__(self, leafsize_max = 100, leafsize_min = 20, leafx=0, leafy=0, mode='mid'):
        self.leafsize_max = leafsize_max
        self.leafsize_min = leafsize_min
        self.leaf_range_x = leafx
        self.leaf_range_y = leafy
        self.root = None
        self.leaflist = []
        self.mode = mode
        self.iteration = 0
        self.verbose = False

    def buildtree(self, pts, verbose=False):
        self.verbose = verbose
        self.root = self.kd(pts, 0)

    def kd(self, pts, level):
        self.iteration += 1
        cmi = pts[:, :2].min(axis = 0)
        cma = pts[:, :2].max(axis = 0)
        rg = cma - cmi
        split = rg.argmax()
        if self.verbose:
            print(self.iteration, split, cmi, cma, pts.shape)
        if pts[:, 3].sum() <= self.leafsize_max or pts.shape[0] < 2 or (rg[0] < self.leaf_range_x and rg[1] < self.leaf_range_y):
            self.leaflist.append(BS2DNode(level, cmi, cma, data=pts[:, 2]))
            return self.leaflist[-1]

        if self.mode == "med":
            pts = pts[pts[:, split].argsort(), :]
            impt = pts.shape[0] // 2
            mpt = (pts[impt, split] + pts[impt + 1, split]) / 2
            if pts.shape[0] < 2 or\
               (pts[impt+1:, 3].sum() < self.leafsize_min and pts[:impt+1, 3].sum() < self.leafsize_max) or\
               (pts[impt+1:, 3].sum() < self.leafsize_max and pts[:impt+1, 3].sum() < self.leafsize_min):
                self.leaflist.append(BS2DNode(level, cmi, cma, data=pts[:, 2]))
                return self.leaflist[-1]
            return BS2DNode(level, cmi, cma,
                        self.kd(pts[:impt+1, :], level+1),
                        self.kd(pts[impt+1:, :], level+1),
                        mpt, split, None)

        mpt = (cmi[split]+cma[split])/2
        sl  = pts[:, split] <= mpt
        sr  = pts[:, split] > mpt
        if (pts[sl, 3].sum() < self.leafsize_min and pts[sr, 3].sum() < self.leafsize_max) or\
           (pts[sl, 3].sum() < self.leafsize_max and pts[sr, 3].sum() < self.leafsize_min):
            self.leaflist.append(BS2DNode(level, cmi, cma, data=pts[:, 2]))
            return self.leaflist[-1]
        return BS2DNode(level, cmi, cma,
                    self.kd(pts[sl, :], level+1),
                    self.kd(pts[sr, :], level+1),
                    mpt, split, None)

    def treeprint(self, order = "pre"):
        if order == "pre":
            self.print_pre(self.root)
            self.print_pre(self.root.right)

    def print_pre(self, node):
        node.printnode()
        if node.data is None:
            self.print_pre(node.left)
            self.print_pre(node.right)
        else:
            return


def traverse_splits_pre(output, node):
    if node.data is not None:
        return
    if node.split == 0:
        output.append([node.midpt, node.midpt,
                       node.coord_min[(node.split+1)%2], node.coord_max[(node.split+1)%2]])
    else:
        output.append([node.coord_min[(node.split+1)%2], node.coord_max[(node.split+1)%2],
                       node.midpt, node.midpt])
    traverse_splits_pre(output, node.left)
    traverse_splits_pre(output, node.right)
