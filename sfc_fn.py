import sys, io, os
import numpy as np

class HCurve:
    """
    H Curve
    https://link.springer.com/chapter/10.1007/BFb0036198
    """

    # Constant
    T=1
    R=2
    B=4
    L=8

    TR = T * 16 + R
    TB = T * 16 + B
    TL = T * 16 + L

    RB = R * 16 + B
    RL = R * 16 + L
    RT = R * 16 + T

    BL = B * 16 + L
    BT = B * 16 + T
    BR = B * 16 + R

    LT = L * 16 + T
    LR = L * 16 + R
    LB = L * 16 + B

    corner_code = {66: 'BR', 132: 'LB', 65: 'BT', 18: 'TR', 129: 'LT',
                   20: 'TB', 36:  'RB', 72: 'BL', 33: 'RT', 24: 'TL'}
    step_dir = {T:[0,1],R:[1,0],B:[0,-1],L:[-1,0]}
    loop_list = [BR, LB, RT, TL]
    grid_list = [BR, LB, BR, LB,
                 BT, TR, LT, TB,
                 BT, RB, BL, TB,
                 RT, TL, RT, TL]
    LUT = {TR: "┗━", TB: "┃ ", TL: "┛ ",
           RB: "┏━", RL: "━━", RT: "┗━",
           BL: "┓ ", BT: "┃ ", BR: "┏━",
           LT: "┛ ", LR: "━━", LB: "┓ "}

    # Functions
    def makeHGrid(self, k):
        size = int(4 * 2**k)
        grid = []
        for y in range(size):
            for x in range(size):
                grid.append(self.grid_list[(y%4)*4 + (x%4)])
        return grid

    def makeLoops(self, k):
        size = int(4 * 2**k)
        grid = []
        for y in range(size):
            for x in range(size):
                grid.append(self.loop_list[(y%2)*2 + (x%2)])
        return grid

    def gridToHCurve(self, grid, x, y, subgridsize):
        gridsize = int(np.sqrt(len(grid)))
        idx = int(x + y * gridsize)
        grid[idx]   = self.BT
        grid[idx+1] = self.TR
        grid[idx+2] = self.LT
        grid[idx+3] = self.TB
        grid[idx+gridsize]   = self.BT
        grid[idx+gridsize+1] = self.RB
        grid[idx+gridsize+2] = self.BL
        grid[idx+gridsize+3] = self.TB

        if subgridsize > 4:
            subgridsize //= 2;
            self.gridToHCurve(grid, x-subgridsize, y-subgridsize, subgridsize)
            self.gridToHCurve(grid, x+subgridsize, y-subgridsize, subgridsize)
            self.gridToHCurve(grid, x-subgridsize, y+subgridsize, subgridsize)
            self.gridToHCurve(grid, x+subgridsize, y+subgridsize, subgridsize)

        return grid;

    def hcurve(self, k):
        size = int(4 * 2 ** k)
        return self.gridToHCurve(self.makeHGrid(k), size // 2 - 2, size // 2 - 1, size // 2)

    def gridToIndices(self, grid):
        size = int(np.sqrt(len(grid)))
        gridToLine = []
        lineToGrid = []
        for i in range(len(grid)):
            gridToLine.append(0)
            lineToGrid.append(0)
        idx = 0
        direction = [0] * len(grid)
        for i in range(len(grid)):
            gridToLine[idx] = i
            lineToGrid[i] = idx
            direction[i] = grid[idx] & 0b1111
            if direction[i] == self.T:
                idx -= size;
            elif direction[i] == self.R:
                idx += 1
            elif direction[i] == self.B:
                idx += size
            else: # direction[i] == self.L:
                idx -= 1
        return (gridToLine, lineToGrid, direction)

    def boxdrawing(self, grid):
        size = int(np.sqrt(len(grid)) )
        output = ""
        for i in range(len(grid)):
            if i % size == 0:
                output += "\n    "
            output += self.LUT[grid[i]];
        return output



class GCurve:
    """
    Peano-Gosper curve
    """

    # Constant
    pattern_A = 'abbaaab'
    pattern_B = 'abbbaab'
    dir_A = [0, 5, 3, 4, 0, 0, 1]
    dir_B = [1, 0, 0, 4, 3, 5, 0]
    alpha = np.arctan((3**0.5) / 5.0)

    k1, k2 = +0.5, +3.0**0.5 / 2.0
    d_cos = {0: +1.0, 1: +k1, 2: -k1, 3: -1.0, 4: -k1, 5: +k1}
    d_sin = {0: +0.0, 1: +k2, 2: +k2, 3: +0.0, 4: -k2, 5: -k2}

    # Function
    def fAddMod6(self, m, d):
        return [(m + e) % 6 for e in d]

    def fRotate(self, c, s, x, y):
        X = [c * xx - s * yy for xx, yy in zip(x ,y)]
        Y = [s * xx + c * yy for xx, yy in zip(x, y)]
        return x, y

    def Gosper(self, max_level = 4, return_all_level = False):
        res = {0: {'s': 7.0**0.5, 't': ['a'], 'd': [0]}}
        for level in range(1, max_level + 1):
            res[level] = {'s': res[level - 1]['s'] / (7.0**.5),
                          't': [], 'd' : []}
            for e, d in zip(res[level - 1]['t'], res[level - 1]['d']):
                res[level]['t'].extend(self.pattern_A if e == 'a' else self.pattern_B)
                res[level]['d'].extend(self.fAddMod6(d, self.dir_A if e == 'a' else self.dir_B))
        if return_all_level:
            return res
        else:
            return res[max_level]

    def gcurve(self, k):
        level = self.Gosper(k, False)
        n = len(level['d']) + 1
        x, y = [0] * n, [0] * n
        for i, d in enumerate(level['d']):
            x[i + 1] = x[i] + self.d_cos[d]
            y[i + 1] = y[i] + self.d_sin[d]
        c, s = np.cos(k * self.alpha), np.sin(k * self.alpha)
        x, y = self.fRotate(c, s, x, y)
        return x, y

    def code_to_curve(self, k, level):
        n = len(level['d']) + 1
        x, y = [0] * n, [0] * n
        for i, d in enumerate(level['d']):
            x[i + 1] = x[i] + self.d_cos[d]
            y[i + 1] = y[i] + self.d_sin[d]
        c, s = np.cos(k * self.alpha), np.sin(k * self.alpha)
        x, y = self.fRotateX(c, s, x, y), self.fRotateY(c, s, x, y)
        return x, y
