import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

# @bootstrap
# put this just on top of recursion function to increase the recursion limit

# rather than return now use yield and when function being called inside itself, use yield before the function name
# example usage:
# @bootstrap
# def rec1(L,k,cur,count):
# 	if count>=100000:
# 		yield float('INF')
# 	if cur+k+1>=len(L)-1:
# 		yield L[cur]+2
# 	if cur in d:
# 		yield d[cur]
# 	ans = float('INF')
# 	mini = float('INF')
# 	for i in range(k+1,0,-1):
# 		if L[cur+i]<mini:
# 			ans = min(ans,1+L[cur]+(yield rec1(L,k,cur+i,count+1)))
# 			mini = L[cur+i]
# 	d[cur] = ans
# 	yield ans
# the limit of recursion on cf is 10**6

# def dijkstra(graph, start):
#     """ 
#         Uses Dijkstra's algortihm to find the shortest path from node start
#         to all other nodes in a directed weighted graph.
#     """
#     n = len(graph)
#     dist, parents = [[float("inf")]*len(graph[0]) for i in range(len(graph))], [[-1]*len(graph[0]) for i in range(len(graph))]
#     dist[start[0]][start[1]] = 0

#     queue = [(0, start)]

#     while queue:
#         path_len, v = heappop(queue)
#         if path_len == dist[v[0]][v[1]]:
#             L1 = [[0,1],[1,0],[-1,0],[0,-1]]
#             for dx,dy in L1:
#                 if not (0<=v[0]+dx<len(dist) and 0<=v[1]+dy<len(dist[0])):
#                     continue
#                 if 1 + path_len < dist[v[0]+dx][v[1]+dy] and graph[v[0]+dx][v[1]+dy]!='#':
#                     dist[v[0]+dx][v[1]+dy], parents[v[0]+dx][v[1]+dy] = 1 + path_len, (dx,dy)
#                     heappush(queue, (1 + path_len, (v[0]+dx,v[1]+dy)))

#     return dist, parents

def bfs(start,end,L2):
    visited = [[False]*len(L2[0]) for i in range(len(L2))]
    Q = deque([(0,start[0],start[1])])
    L = [(0,1,'R'),(1,0,'D'),(-1,0,'U'),(0,-1,'L')]    

    go = [[-1 for i in range(len(L2[0]))] for j in range(len(L2))]
    
    def ispossible(x,y):
        return 0<=x<len(L2) and 0<=y<len(L2[0]) and L2[x][y]!='#' and not visited[x][y]
    ans = -1
    while Q:
        dist,x,y = Q.pop()
        for dx,dy,type in L:
            if ispossible(x+dx,y+dy):
                k1 = x+dx
                k2 = y+dy
                go[k1][k2]= type
                visited[x+dx][y+dy] = True
                Q.appendleft((dist+1,x+dx,y+dy))
        if visited[end[0]][end[1]]:
            ans = dist+1
            break
    go[start[0]][start[1]] = -2
    if ans!=-1:
        # print(go)
        d = {'U':(1,0),'D':(-1,0),'R':(0,-1),'L':(0,1)}
        print("YES")
        print(ans)
        # for i in go:
        #     print(*i)
        x,y = end
        ans = []
        while go[x][y]!=-2:
            k = go[x][y]
            ans.append(k)
            x += d[k][0]
            y += d[k][1]
        print(''.join(ans[::-1]))
    else:
        print("NO")
def solve():
    L = list(map(int, sys.stdin.readline().split()))
    #L1 = list(map(int, sys.stdin.readline().split()))
    L2 = []
    for i in range(L[0]):
        st = sys.stdin.readline().strip()
        L2.append(list(st))
        for j in range(len(st)):
            if st[j]=='A':
                x,y = i,j
            elif st[j]=='B':
                x1,y1 = i,j
        # if 'A' in st:
        #     x,y = i,st.index('A')
        # if 'B' in st:
        #     x1,y1 = i,st.index('B')
    (bfs((x,y),(x1,y1),L2))

solve()