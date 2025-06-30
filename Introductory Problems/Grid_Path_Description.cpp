// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vector<int>>;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<typename T>
class is_printable {
    template<typename U> static auto test(int) -> decltype(cout << declval<U>(), true_type{});
    template<typename> static auto test(...) -> false_type;
public: static constexpr bool value = decltype(test<T>(0))::value;
};
template<typename T> enable_if_t<is_printable<T>::value> __print(const T& val) { cout << val; }
template<typename T1, typename T2> void __print(const pair<T1, T2>& p) {
    cout << "("; __print(p.first); cout << ":"; __print(p.second); cout << ")";
}
template<typename T> enable_if_t<!is_printable<T>::value> __print(const T& container) {
    cout << "{"; bool first = true;
    for (const auto& x : container) { if (!first) cout << ", "; __print(x); first = false; }
    cout << "}";
}
void print() { cout << "\n"; }
template<typename T, typename... Args>
void print(const T& t, const Args&... rest) {
    __print(t); if constexpr (sizeof...(rest)) cout << " "; print(rest...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, SortedList: sortl
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

const int n = 7;
string st;
bool visited[7][7];

int dirs[4][2] = {{0,1}, {0,-1}, {1,0}, {-1,0}};
char dirChars[4] = {'R', 'L', 'D', 'U'};

int rec(int i, int x, int y) {
    if (i == (int)st.length()) return (x == 6 && y == 0);
    if (visited[x][y] || (x == 6 && y == 0)) return 0;

    bool vis1[4] = {true, true, true, true};
    for (int d = 0; d < 4; ++d) {
        int nx = x + dirs[d][0], ny = y + dirs[d][1];
        if (nx >= 0 && nx < 7 && ny >= 0 && ny < 7) {
            vis1[d] = visited[nx][ny];
        }
    }

    if (!vis1[2] && !vis1[3] && vis1[0] && vis1[1]) return 0;
    if (!vis1[0] && !vis1[1] && vis1[2] && vis1[3]) return 0;
    if (x-1 >= 0 && y+1 < 7 && visited[x-1][y+1] && !vis1[0] && !vis1[3]) return 0;
    if (x+1 < 7 && y+1 < 7 && visited[x+1][y+1] && !vis1[0] && !vis1[2]) return 0;
    if (x-1 >= 0 && y-1 >= 0 && visited[x-1][y-1] && !vis1[1] && !vis1[3]) return 0;
    if (x+1 < 7 && y-1 >= 0 && visited[x+1][y-1] && !vis1[1] && !vis1[2]) return 0;

    visited[x][y] = true;
    int ans = 0;
    if (st[i] == '?') {
        for (int d = 0; d < 4; ++d) {
            int nx = x + dirs[d][0], ny = y + dirs[d][1];
            if (nx >= 0 && nx < 7 && ny >= 0 && ny < 7) {
                ans += rec(i + 1, nx, ny);
            }
        }
    } 
    else {
        int dx = 0, dy = 0;
        if (st[i] == 'D') dx = 1;
        else if (st[i] == 'U') dx = -1;
        else if (st[i] == 'R') dy = 1;
        else if (st[i] == 'L') dy = -1;
        int nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < 7 && ny >= 0 && ny < 7) {
            ans += rec(i + 1, nx, ny);
        }
    }
    visited[x][y] = false;
    return ans;
}

void solve() {
    st = SI();
    print(rec(0,0,0));
}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}