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
 
# def rec(i,j,ist,visited,st):
#     print(i,j,ist)
#     if ist==len(st) and i==6 and j==6:
#         return 1
#     elif ist==len(st):
#         return 0
#     visited[i][j] = True
#     L = [[0,1],[0,-1],[1,0],[-1,0]]
#     ans = 0
    
#     def ispossible(x,y):
#         return (0<=x<7 and 0<=y<7 and not visited[x][y])
    
#     def ispossible1(x,y):
#         return (0<=x<7 and 0<=y<7)
    
#     def isvalid(x,y):
#         vis = [True]*4
#         for k in range(4):
#             if ispossible1(x + L[k][0],y + L[k][1]):
#                 vis[k] = visited[x + L[k][0]][y + L[k][1]]
#         if not vis[2] and not vis[3] and vis[0] and vis[1]:
#             return False
#         if not vis[0] and not vis[1] and vis[2] and vis[3]:
#             return False
#         if ispossible1(x - 1,y + 1) and visited[x - 1][y + 1]:
#             if not vis[0] and not vis[3]:
#                 return False
#         if ispossible1(x+1,y+1) and visited[x + 1][y + 1]:
#             if not vis[0] and not vis[2]:
#                 return False
#         if ispossible1(x-1,y-1) and visited[x - 1][y - 1] == 1:
#             if not vis[1] and not vis[3]:
#                 return False
#         if ispossible1(x+1,y-1) and visited[x + 1][y - 1] == 1:
#             if not vis[1] and not vis[2]:
#                 return False
#         return True
#     if st[ist]!='?':
#         if st[ist]=='R':
#             dx = 0
#             dy = 1
#             if ispossible(i+dx,j+dy) and isvalid(i+dx,j+dy):
#                 ans += rec(i+dx,j+dy,ist+1,visited,st)
#         elif st[ist]=='L':
#             dx = 0
#             dy = -1
#             if ispossible(i+dx,j+dy) and isvalid(i+dx,j+dy):
#                 ans += rec(i+dx,j+dy,ist+1,visited,st)
#         elif st[ist]=='D':
#             dx = 1
#             dy = 0
#             if ispossible(i+dx,j+dy) and isvalid(i+dx,j+dy):
#                 ans += rec(i+dx,j+dy,ist+1,visited,st)
#         else:
#             dx = -1
#             dy = 0
#             if ispossible(i+dx,j+dy) and isvalid(i+dx,j+dy):
#                 ans += rec(i+dx,j+dy,ist+1,visited,st)
#     else:
#         for dx,dy in L:
#             if ispossible(i+dx,j+dy) and isvalid(i+dx,j+dy):
#                 ans += rec(i+dx,j+dy,ist+1,visited,st)
    
#     visited[i][j] = False
 
#     return ans
 
@bootstrap
def rec(x, y, pos,visited,str_path):
    L = [[0,1],[0,-1],[1,0],[-1,0]]
    if pos == len(str_path):
        yield 1 if x == 6 and y == 0 else 0
    if x == 6 and y == 0:
        yield 0
    if visited[x][y]:
        yield 0
    def func(x):
        return 0<=x<7
    vis1 = [-1]*4
    for k in range(4):
        if func(x + L[k][0]) and func(y + L[k][1]):
            vis1[k] = visited[x + L[k][0]][y + L[k][1]]
    if not vis1[2] and not vis1[3] and vis1[0] and vis1[1]:
        yield 0
    if not vis1[0] and not vis1[1] and vis1[2] and vis1[3]:
        yield 0
    if func(x - 1) and func(y + 1) and visited[x - 1][y + 1] == 1:
        if not vis1[0] and not vis1[3]:
            yield 0
    if func(x + 1) and func(y + 1) and visited[x + 1][y + 1] == 1:
        if not vis1[0] and not vis1[2]:
            yield 0
    if func(x - 1) and func(y - 1) and visited[x - 1][y - 1] == 1:
        if not vis1[1] and not vis1[3]:
            yield 0
    if func(x + 1) and func(y - 1) and visited[x + 1][y - 1] == 1:
        if not vis1[1] and not vis1[2]:
            yield 0
    visited[x][y] = 1
    ans = 0
    if str_path[pos] == '?':
        for k in range(4):
            if func(x + L[k][0]) and func(y + L[k][1]):
                ans += (yield rec(x + L[k][0], y + L[k][1], pos + 1,visited,str_path))
    elif str_path[pos] == 'R' and y + 1 < 7:
        ans += (yield rec(x, y + 1,pos+1,visited,str_path))
    elif str_path[pos] == 'L' and y - 1 >= 0:
        ans += (yield rec(x, y - 1,pos+1,visited,str_path))
    elif str_path[pos] == 'U' and x - 1 >= 0:
        ans += (yield rec(x - 1, y,pos+1,visited,str_path))
    elif str_path[pos] == 'D' and x + 1 < 7:
        ans += (yield rec(x + 1, y,pos+1,visited,str_path))
    visited[x][y] = 0
    yield ans
 
def solve():
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    vis1 = [[False for i in range(7)] for j in range(7)]
    print(rec(0,0,0,vis1,st))
 
solve()