import sys
def solve():
    n,X = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    # L1.sort()
    dp = [0]*(X+1)
    dp[0] = 1
    MOD = 10**9 + 7
    for i in range(X+1):
        if dp[i]:
            dp[i] %= MOD
            for j in L1:
                if i+j<=X:
                    dp[i+j] += dp[i]
    print(dp[X])
    #st = sys.stdin.readline().strip()
solve()