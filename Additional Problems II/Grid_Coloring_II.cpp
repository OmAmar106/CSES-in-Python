// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define int long long
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vi>;
using viii = vector<vii>;
using ll = long long;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#ifdef LOCAL
#define dbg(x...) cout<<"["<<#x<<"] = "; print(x);
#else
#define dbg(x...)
#endif

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<class T> bool chmin(T& a,const T& b){ if(b<a){ a=b; return 1;} return 0; }
template<class T> bool chmax(T& a,const T& b){ if(a<b){ a=b; return 1;} return 0;}
template<class T> using minpq = priority_queue<T,vector<T>,greater<T>>;
template<class T> using maxpq = priority_queue<T>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<class T> void _print(const T &x){ cout << x; }
template<class T,class U> void _print(const pair<T,U> &p){cout<<"(";_print(p.first);cout<<",";_print(p.second);cout<<")";}
template<class T>
void _print(const vector<T> &v){
    cout<<"[";bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"]";
}
template<class T>
void _print(const set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T>
void _print(const unordered_set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const unordered_map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
void print(){ cout<<"\n"; }
template<class T,class... Args>
void print(const T& a,const Args&... args){
    _print(a);
    if constexpr(sizeof...(args)) cout<<" ";
    print(args...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
int rnd(int l,int r){return uniform_int_distribution<int>(l,r)(rng);}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Rollback/Par DSU: rbdsu, treap: treap, graphflow(mat_match): graphflow, Persistent Seg Tree: perseg
// Segment Tree(Nodes): SegmentNode, HLD: hld, fwht: fwht
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

vector<int> bellman_ford(int n, vector<tuple<int, int, int>> &edges, int start){
    vector<int> dist(n, INF), pred(n, -1);
    dist[start] = 0;
    for (int i = 0; i < n - 1; i++) {
        for (int i = 0; i < edges.size(); i++) {
        int u = get<0>(edges[i]);
        int v = get<1>(edges[i]);
        int d = get<2>(edges[i]);
            if (dist[u] != INF && dist[u] + d < dist[v]) {
                dist[v] = dist[u] + d;
                pred[v] = u;
            }
        }
    }
    return dist;
}

class BinaryLift {
public:
    int n, L;
    vector<int> depth;
    vector<vector<int>> parent;

    BinaryLift(vector<vector<int>>& graph, int root = 0) {
        n = graph.size();
        L = 32 - __builtin_clz(n);
        parent.assign(L, vector<int>(n, -1));
        depth.assign(n, -1);
        queue<int> bfs;
        bfs.push(root);
        depth[root] = 0;
        while (!bfs.empty()) {
            int node = bfs.front(); bfs.pop();
            for (int nei : graph[node]) {
                if (depth[nei] == -1) {
                    parent[0][nei] = node;
                    depth[nei] = depth[node] + 1;
                    bfs.push(nei);
                }
            }
        }
        for (int i = 1; i < L; i++) {
            for (int v = 0; v < n; v++) {
                if (parent[i - 1][v] != -1)
                    parent[i][v] = parent[i - 1][parent[i - 1][v]];
            }
        }
    }

    int lca(int a, int b) {
        if (depth[a] < depth[b]) swap(a, b);
        int d = depth[a] - depth[b];
        for (int i = 0; i < L; i++) {
            if (d & (1LL << i)) a = parent[i][a];
        }
        if (a == b) return a;
        for (int i = L - 1; i >= 0; i--) {
            if (parent[i][a] != parent[i][b]) {
                a = parent[i][a];
                b = parent[i][b];
            }
        }
        return parent[0][a];
    }

    int distance(int a, int b) {
        return depth[a] + depth[b] - 2 * depth[lca(a, b)];
    }

    int kth_ancestor(int a, int k) {
        if (depth[a] < k) return -1;
        for (int i = 0; i < L; i++) {
            if (k & (1LL << i)) a = parent[i][a];
        }
        return a;
    }
};

vector<int> kahn(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> indeg(n, 0), res;
    for (int i = 0; i < n; i++)
        for (int e : graph[i])
            indeg[e]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indeg[i] == 0) q.push(i);
    while (!q.empty()) {
        int node = q.front(); q.pop();
        res.push_back(node);
        for (int e : graph[node]) {
            indeg[e]--;
            if (indeg[e] == 0) q.push(e);
        }
    }
    return res.size() == n ? res : vector<int>();
}

vector<pair<int, int>> dfs(vector<vector<int>>& graph) {
    int n = graph.size(), time = 0;
    vector<pair<int, int>> starttime(n, {0, 0});
    stack<tuple<int, int, int>> stack;
    stack.emplace(0, -1, 0);
    while (!stack.empty()) {
        int cur, prev, state;
        cur = get<0>(stack.top());
        prev = get<1>(stack.top());
        state = get<2>(stack.top());
        stack.pop();
        if (state == 0) {
            starttime[cur].first = time++;
            stack.emplace(cur, prev, 1);
            for (int neighbor : graph[cur]) {
                if (neighbor == prev) continue;
                stack.emplace(neighbor, cur, 0);
            }
        } else {
            starttime[cur].second = time;
        }
    }
    return starttime;
}

vector<int> euler_path(unordered_map<int, multiset<int>>& d) {
    vector<int> ans;
    stack<int> start;
    start.push(1);
    while (!start.empty()) {
        int cur = start.top();
        if (d[cur].empty()) {
            ans.push_back(cur);
            start.pop();
        } else {
            int k1 = *d[cur].begin();
            d[cur].erase(d[cur].begin());
            d[k1].erase(d[k1].find(cur));
            start.push(k1);
        }
    }
    return ans;
}

class TwoSat {
public:
    int n;
    vector<vector<int>> graph;

    TwoSat(int n) : n(n), graph(2 * n) {}

    int negate(int x) {
        return x < n ? x + n : x - n;
    }

    void _imply(int x, int y) {
        graph[x].push_back(y);
        graph[negate(y)].push_back(negate(x));
    }

    void either(int x, int y) {
        _imply(negate(x), y);
        _imply(negate(y), x);
    }

    void set(int x) {
        _imply(negate(x), x);
    }

    vector<vector<int>> find_SCC() {
        int n = graph.size();
        vector<int> order, comp(n, -1), low(n), depth(n, -1);
        vector<vector<int>> SCC;
        stack<int> st;
        int timer = 0;

        function<void(int)> dfs = [&](int v) {
            low[v] = depth[v] = timer++;
            st.push(v);
            for (int u : graph[v]) {
                if (depth[u] == -1) dfs(u);
                if (comp[u] == -1) low[v] = min(low[v], low[u]);
            }
            if (low[v] == depth[v]) {
                SCC.emplace_back();
                while (true) {
                    int u = st.top(); st.pop();
                    comp[u] = SCC.size() - 1;
                    SCC.back().push_back(u);
                    if (u == v) break;
                }
            }
        };

        for (int i = 0; i < n; i++) {
            if (depth[i] == -1) dfs(i);
        }
        return SCC;
    }

    pair<bool, vector<int>> solve() {
        auto SCC = find_SCC();
        vector<int> order(2 * n, 0);
        for (int i = 0; i < SCC.size(); i++)
            for (int x : SCC[i])
                order[x] = i;
        vector<int> res(n);
        for (int i = 0; i < n; i++) {
            if (order[i] == order[negate(i)])
                return {false, {}};
            res[i] = order[i] < order[negate(i)];
        }
        return {true, res};
    }
};

void solve() {
    int n,m;
    cin>>n>>m;

    vector<char> L;

    for(int i=0;i<n;i++){
        string s;
        cin>>s;
        for(auto it:s){
            L.pb(it);
        }
    }
    
    TwoSat sat = TwoSat(3*n*m);

    auto f2 = [](int ind,vector<char> &L, TwoSat &sat){
        int cur1 = L[ind]-'A';
        ind *= 3;
        if(cur1==0){
            sat._imply(ind+1,sat.negate(ind+2));
            sat._imply(sat.negate(ind+2),ind+1);
        }
        else if(cur1==1){
            sat._imply(ind,sat.negate(ind+2));
            sat._imply(sat.negate(ind+2),ind);
        }
        else{
            sat._imply(ind+1,sat.negate(ind));
            sat._imply(sat.negate(ind),ind+1);
        }
    };

    auto medge = [](int pos1,int pos2,vector<char> &L, TwoSat &sat){
        for(int j=0;j<3;j++){
            sat._imply(3*pos1+j,sat.negate(3*pos2+j));
        }
    };

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            int cur = m*i+j;
            if(i+1<n){
                medge(cur,cur+m,L,sat);
            }
            if(j+1<m){
                medge(cur,cur+1,L,sat);
            }
            f2(cur,L,sat);
        }
    }

    auto [f,ans] = sat.solve();

    if(!f){
        print("IMPOSSIBLE");
        return;
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(ans[3*(i*m+j)]){
                cout<<'A';
            }
            else if(ans[3*(i*m+j)+1]){
                cout<<'B';
            }
            else{
                cout<<'C';
            }
        }
        cout<<endl;
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}