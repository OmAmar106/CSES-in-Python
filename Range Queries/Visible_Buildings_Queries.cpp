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

void solve() {
    int n,q;
    cin>>n>>q;

    vi L1 = LII(n);

    vi st;
    vi next(n,n);

    for(int i=n-1;i>=0;i--){
        while(!st.empty() && L1[st.back()]<=L1[i]){
            st.pop_back();
        }
        if(!st.empty()){
            next[i] = st.back();
        }
        st.pb(i);
    }

    vector<tuple<int,int,int>> L;
    range(i,q){
        int a,b;
        cin>>a>>b;
        a--;b--;
        L.emplace_back(a,b,i);
    }

    sort(all(L));

    vi ans;
    int i = 0;
    while(next[i]!=n){
        ans.pb(i);
        i = next[i];
    }
    ans.pb(i);
    reverse(all(ans));

    vi fans(q,0);

    for(auto &[l,r,index]:L){
        // print(ans);
        while(ans.back()<l){
            int t = ans.back();
            // print(ans);
            ans.pop_back();
            // print(ans);
            vi ans2 = {t+1};
            if(!ans.empty() && t+1==ans.back()){
                continue;
            }
            while(next[ans2.back()]!=n && (ans.empty() || next[ans2.back()]!=ans.back())){
                ans2.pb(next[ans2.back()]);
            }
            // print(ans2);
            for(int i=len(ans2)-1;i>=0;i--){
                ans.pb(ans2[i]);
            }
        }

        int start = 0;
        int end = len(ans)-1;

        // print(ans);
        while(start<=end){
            int mid = (start+end)/2;
            if(ans[mid]<=r){
                end = mid-1;
            }
            else{
                start = mid+1;
            }
        }

        fans[index] = len(ans)-end-1;

    }
    
    for(auto it:fans){
        cout<<it<<endl;
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}