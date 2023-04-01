import numpy as np

class brt_node:

    def __init__(self, idx, size = 1, p = None, c = [], ll = None, score = None) -> None:
        self.id = idx
        self.size = size
        self.p = p
        self.c = c
        self.ll = ll
        self.score = score
        if self.score is None:
            self.score = self.ll

class BRT:

    def __init__(self, mtx, labels = None, gamma=.75,  verbose = False, debug = 0) -> None:
        self.N, self.M = mtx.shape
        self.next_id = 0
        self.gamma = gamma
        self.tree = {}
        self.frontier = []
        self.verbose = verbose
        self.debug = debug
        ct = mtx.sum(axis = 1)
        if labels is None:
            self.labels = np.arange(self.N)
            self.next_id = self.N
            self.sstats = np.vstack((mtx, mtx))
            for i in range(self.N):
                self.tree[i] = brt_node(i, size = ct[i], p = self.next_id, c = None, ll = self.mle_multi(i))
                self.tree[self.next_id] = brt_node(self.next_id, size = ct[i], c = [i], ll = self.tree[i].ll)
                self.frontier.append(self.next_id)
                self.next_id += 1
        else:
            label_list = {x:i for i,x in enumerate(set(labels))}
            self.labels = np.array([label_list[x] for x in labels] )
            self.next_id = len(label_list)
            self.sstats = np.zeros((len(label_list), self.M))
            for k in label_list.values():
                indx = list(np.arange(self.N)[self.labels==k] )
                self.sstats[k, :] = mtx[indx, :].sum(axis = 0)
                self.tree[k] = brt_node(k, size=ct[indx].sum(), c = [k], ll = self.mle_multi(k))
                self.frontier.append(k)

    def mle_multi(self, k):
        p = self.sstats[k, :] + .1
        p /= p.sum()
        return (np.log(p) * self.sstats[k, :]).sum()

    def join_score(self, k, l):
        return self.tree[k].score + self.tree[l].score, (1 - self.gamma)

    def absorb_score(self, k, l):
        s = self.tree[k].score
        for c in self.tree[l].c:
            s += self.tree[c].score
        return s,  (1 - self.gamma) ** len(self.tree[l].c)

    def collapse_score(self, k, l):
        s = 0
        for c in self.tree[k].c:
            s += self.tree[c].score
        for c in self.tree[l].c:
            s += self.tree[c].score
        return s, (1 - self.gamma) ** (len(self.tree[k].c) + len(self.tree[l].c) - 1)

    def join(self, k, l, s): # join k and l, add a new node
        self.tree[self.next_id] = brt_node(self.next_id, p = None, c = [k, l])
        self.tree[self.next_id].score = s
        self.tree[k].p = self.next_id
        self.tree[l].p = self.next_id
        self.sstats = np.vstack((self.sstats, self.sstats[k, :] + self.sstats[l, :]))
        self.tree[self.next_id].ll = self.mle_multi(self.next_id)
        self.tree[self.next_id].size = self.tree[k].size + self.tree[l].size
        self.frontier.remove(k)
        self.frontier.remove(l)
        self.frontier.append(self.next_id)
        self.next_id += 1

    def absorb(self, k, l, s): # absorb l into k
        self.tree[k].c.append(l)
        self.tree[l].p = k
        self.tree[k].score = s
        self.sstats[k, :] += self.sstats[l, :]
        self.tree[k].ll = self.mle_multi(k)
        self.tree[k].size += self.tree[l].size
        self.frontier.remove(l)

    def collapse(self, k, l, s): # collapse k and l
        # self.tree[self.next_id] = brt_node(self.next_id, p = None, c = self.tree[k].c + self.tree[l].c)
        # for c in self.tree[l].c:
        #     self.tree[c].p = self.next_id
        # for c in self.tree[k].c:
        #     self.tree[c].p = self.next_id
        # self.tree[self.next_id].score = s
        # self.sstats = np.vstack((self.sstats, self.sstats[k, :] + self.sstats[l, :] ))
        # self.tree[self.next_id].ll = self.mle_multi(self.next_id)
        # self.tree[self.next_id].size = self.tree[k].size + self.tree[l].size
        # self.frontier.remove(l)
        # self.frontier.remove(k)
        # del self.tree[l]
        # del self.tree[k]
        self.tree[k] = brt_node(self.next_id, p = None, c = self.tree[k].c + self.tree[l].c)
        self.tree[k].score = s
        for c in self.tree[l].c:
            self.tree[c].p = k
        self.sstats[k, :] += self.sstats[l, :]
        self.tree[k].ll = self.mle_multi(k)
        self.tree[k].size += self.tree[l].size
        self.frontier.remove(l)
        del self.tree[l]

    def select_pair(self):
        best_score = -np.inf
        best_pair = None
        best_move = -1
        for i, k in enumerate(self.frontier[:-1]):
            for j, l in enumerate(self.frontier[i+1:]):
                p = self.sstats[k, :] + self.sstats[l, :] + .1
                p /= p.sum()
                ll = (np.log(p) * (self.sstats[k, :] + self.sstats[l, :])).sum()
                subtree_ll = [self.join_score(k, l), self.absorb_score(k, l), self.absorb_score(l, k), self.collapse_score(k, l)]
                candidate_score = [x[0] * x[1] + ll * (1-x[1]) for x in subtree_ll]
                move = np.argmax(candidate_score)
                score = candidate_score[move] - self.tree[k].score - self.tree[l].score
                if score > best_score:
                    best_score = score
                    best_pair = (k, l)
                    best_move = move
        return best_pair[0], best_pair[1], best_move, best_score

    def build_tree(self):
        while len(self.frontier) > 1:
            k, l, move, s = self.select_pair()
            if self.verbose:
                print("move: ", move, " (k, l): ", (k,l))
            if move == 0:
                self.join(k, l, s)
            elif move == 1:
                self.absorb(k, l, s)
            elif move == 2:
                self.absorb(l, k, s)
            else:
                self.collapse(k, l, s)
        self.root = [k for k,v in self.tree.items() if v.p is None][0]

    def get_edges(self):
        edges = []
        for k, v in self.tree.items():
            if v.p is not None:
                edges.append((v.p, k))
        return edges

    def condense_edges(self):
        to_rm = []
        for k,v in self.tree.items():
            if v.p is None:
                continue
            if v.c is None:
                continue
            if len(v.c) == 1:
                self.tree[v.c[0]].p = v.p
                self.tree[v.p].c.remove(k)
                self.tree[v.p].c.append(v.c[0])
                to_rm.append(k)
        for k in to_rm:
            del self.tree[k]

    def depth_first_travel(self, k):
        if self.tree[k].c is None:
            return [k]
        else:
            s = []
            for c in self.tree[k].c:
                s += self.depth_first_travel(c)
            return s
