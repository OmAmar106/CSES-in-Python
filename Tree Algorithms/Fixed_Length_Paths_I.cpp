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

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

int ctz(int x) { return __builtin_ctz(x); }
int forbidden[200001];

pair<vector<vector<int>>, int> shallowest_decomposition_tree(const vector<vector<int>>& graph, int root = 0) {
    int n = graph.size();
    vector<vector<int>> decomposition_tree(n);
    vector<vector<int>> stacks(32);

    auto extract_chain = [&](int labels, int u) {
        while (labels) {
            int label = 31 - __builtin_clz(labels);
            labels ^= (1 << label);
            int v = stacks[label].back();
            stacks[label].pop_back();
            decomposition_tree[u].push_back(v);
            u = v;
        }
    };

    vector<int> dfs;
    dfs.push_back(root);

    while (!dfs.empty()) {
        int u = dfs.back(); dfs.pop_back();
        if (u >= 0) {
            forbidden[u] = -1;
            dfs.push_back(~u);
            for (int v : graph[u])
                if (!forbidden[v]) dfs.push_back(v);
        } else {
            u = ~u;
            int forbidden_once = 0, forbidden_twice = 0;
            for (int v : graph[u]) {
                forbidden_twice |= forbidden_once & (forbidden[v] + 1);
                forbidden_once |= forbidden[v] + 1;
            }
            forbidden[u] = forbidden_once | ((1 << (32 - __builtin_clz(forbidden_twice))) - 1);
            int label_u = ctz(forbidden[u] + 1);
            stacks[label_u].push_back(u);
            for (int v : graph[u])
                extract_chain((forbidden[v] + 1) & ((1 << label_u) - 1), u);
        }
    }

    int max_label = 31 - __builtin_clz(forbidden[root] + 1);
    int decomposition_root = stacks[max_label].back();
    stacks[max_label].pop_back();
    extract_chain((forbidden[root] + 1) & ((1 << max_label) - 1), decomposition_root);

    return {decomposition_tree, decomposition_root};
}

int dead[200001];
int depth[200001];
int parent[200001];
int n,k,u,v;

void solve() {
    cin>>n>>k;
    vii d(n);
    for(int i=0;i<n-1;i++){
        cin>>u>>v;
        u--;v--;
        d[u].pb(v);
        d[v].pb(u);
    }

    auto [L,root] = shallowest_decomposition_tree(d);

    vi stack = {root};
    int ans = 0;

    while(!stack.empty()){
        int start = stack.back();
        stack.pop_back();
        unordered_map<int,int> count;
        dead[start] = true;
        for(auto j:L[start]){
            stack.pb(j);
        }
        for(auto j:d[start]){
            if(dead[j]){
                continue;
            }
            vector<int> st = {j};
            depth[j] = 1;
            vi L9;
            parent[j] = -1;
            while(!st.empty()){
                int start1 = st.back();
                st.pop_back();
                if(depth[start1]==k){
                    ans++;
                    continue;
                }
                ans += count[k-depth[start1]];
                L9.pb(depth[start1]);
                for(auto child:d[start1]){
                    if(child!=parent[start1] && !dead[child]){
                        parent[child] = start1;
                        depth[child] = depth[start1]+1;
                        st.pb(child);
                    }
                }
            }
            for(auto it:L9){
                count[it] += 1;
            }
        }
    }
    print(ans);
}

int32_t main() {
    fastio
    solve();
    return 0;
}