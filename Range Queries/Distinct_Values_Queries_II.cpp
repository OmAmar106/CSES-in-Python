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
// #define f first
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

template <typename T>
class SortedList {
    ordered_multiset<T> os;
    int uid = 0;
public:
    void insert(T x) {
        os.insert({x, uid++});
    }
    void erase_one(T x) {
        auto it = os.lower_bound({x, 0});
        if (it != os.end() && it->first == x)
            os.erase(it);
    }
    int index(T x) {
        return os.order_of_key({x, 0});
    }
    int count(T x) {
        return os.order_of_key({x, INT_MAX}) - os.order_of_key({x, 0});
    }
    T operator[](int k) {
        return os.find_by_order(k)->first;
    }
    int size() const {
        return os.size();
    }
    void print() {
        for (int i = 0; i < size(); i++) std::cout << (*this)[i] << " ";
        std::cout << "\n";
    }
};

class SegmentTree {
public:
    int n;
    vector<int> tree;

    static int func(int a, int b) {
        return min(a, b);
    }

    SegmentTree(const vector<int>& data) {
        n = data.size();
        tree.resize(2 * n);
        build(data);
    }

    void build(const vector<int>& data) {
        for (int i = 0; i < n; ++i) {
            tree[n + i] = data[i];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = func(tree[i * 2], tree[i * 2 + 1]);
        }
    }

    void update(int pos, int value) {
        pos += n;
        tree[pos] = value;
        while (pos > 1) {
            pos /= 2;
            tree[pos] = func(tree[2 * pos], tree[2 * pos + 1]);
        }
    }

    int query(int left, int right) {
        left += n;
        right += n;
        int max_val = LLONG_MAX;
        while (left < right) {
            if (left % 2 == 1) {
                max_val = func(max_val, tree[left]);
                left++;
            }
            if (right % 2 == 1) {
                right--;
                max_val = func(max_val, tree[right]);
            }
            left /= 2;
            right /= 2;
        }
        return max_val;
    }
};

void solve() {
    int n,q;
    cin>>n>>q;

    vi L = LII(n);
    map<int,SortedList<int>> d;
    vi L1(n,n);

    for(int i=n-1;i>=0;i--){
        if(d[L[i]].size()){
            L1[i] = d[L[i]][0];
        }
        d[L[i]].insert(i);
    }

    SegmentTree seg(L1);

    while(q--){
        int t,a,b;
        cin>>t>>a>>b;
        a--;
        if(t==1){
            if(L[a]==b){
                continue;
            }
            int f = d[L[a]].index(a);
            if(f-1>=0){
                seg.update(d[L[a]][f-1],(f+1<len(d[L[a]]))?d[L[a]][f+1]:n);
            }
            d[L[a]].erase_one(a);
            L[a] = b;
            d[L[a]].insert(a);
            f = d[L[a]].index(a)+1;
            seg.update(d[L[a]][f-1],(f<len(d[L[a]]))?d[L[a]][f]:n);
            if(f-2>=0){
                seg.update(d[L[a]][f-2],d[L[a]][f-1]);
            }
        }
        else{
            if(seg.query(a,b)>b-1){
                print("YES");
            }
            else{
                print("NO");
            }
        }
        // for(int i=0;i<n;i++){
        //     cout<<seg.query(i,i+1)<<" ";
        // }
        // cout<<endl;
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}