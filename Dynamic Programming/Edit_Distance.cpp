// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define print(dp) for (auto it : dp){cout<<it<<" ";}cout<<endl;
#define len(dp) dp.size()
#define printf(x) cout << x << endl;
#define printm(map) cout<<"{";for(auto it: map){cout<<it.first<<":"<<it.second<<",";};cout<<"}"<<endl;

// const int MOD = 998244353;
const int MOD = 1e9 + 7;

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return a / gcd(a, b) * b;
}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void solve() {
    string s;
    string s1;
    cin>>s;
    cin>>s1;

    int dp[len(s)][len(s1)];
    if(s[0]==s1[0]){
        dp[0][0] = 0;
    }
    else{
        dp[0][0] = 1;
    }
    for(int i=0;i<len(s);i++){
        for(int j=0;j<len(s1);j++){
            if((i-1>=0) || (j-1>=0)){
                dp[i][j] = INF;
                if(s[i]==s1[j] && (i>=1) &&(j>=1)){
                    dp[i][j] = min(dp[i][j],dp[i-1][j-1]);
                }
                else if((i>=1) && (j>=1)){
                    dp[i][j] = min(dp[i][j],dp[i-1][j-1]+1);
                }
                else if(s[i]==s1[j]){
                    dp[i][j] = min(dp[i][j],max(i,j));
                }
                if(i>=1){
                    dp[i][j] = min(dp[i][j],dp[i-1][j]+1);
                }
                if(j>=1){
                    dp[i][j] = min(dp[i][j],dp[i][j-1]+1);
                }
            }
        }
    }
    printf(dp[len(s)-1][len(s1)-1]);
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}