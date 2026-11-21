#include "output.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <limits>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <mutex>
#include <thread>
#include <unordered_map>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace std::chrono;
using std::vector;
template<class V> using Dict = std::unordered_map<Text,V>;
constexpr int bins = 64;
constexpr I64 day_ms = 86400000, week = 7*day_ms, bad = INT64_MIN;
int threads = 8;

template<class F> void parallel(size_t tasks, F work) {
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex guard;
    {
        vector<std::jthread> pool;
        for (size_t i=0; i<std::min<size_t>(tasks,threads); ++i)
            pool.emplace_back([&] {
                try {
                    for (size_t n; (n=next.fetch_add(1))<tasks;) work(n);
                } catch (...) {
                    std::lock_guard lock(guard);
                    error=std::current_exception();
                    next.store(tasks);
                }
            });
    }
    if (error) std::rethrow_exception(error);
}

std::string read(const fs::path& path) {
    std::ifstream stream(path,std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    std::string data(fs::file_size(path),0);
    stream.read(data.data(),data.size());
    return data;
}
template<size_t N> bool fields(Text line, std::array<Text,N>& out) {
    for (size_t i=0;i<N;++i) {
        auto comma=line.find(',');
        if ((comma==Text::npos)!=(i+1==N)) return false;
        Text value=line.substr(0,comma);
        if (value.starts_with('"') && value.ends_with('"') && value.size()>=2)
            value=value.substr(1,value.size()-2);
        if (value.find('"')!=Text::npos) return false;
        out[i]=value;
        if (comma!=Text::npos) line.remove_prefix(comma+1);
    }
    return true;
}
#if defined(__AVX2__) && defined(__BMI2__)
struct CsvMask {
    uint64_t commas,quotes,newlines;
};
inline CsvMask classify64(const char* p) {
    auto low=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    auto high=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+32));
    auto matches=[&](char c) {
        auto target=_mm256_set1_epi8(c);
        uint32_t a=_mm256_movemask_epi8(_mm256_cmpeq_epi8(low,target));
        uint32_t b=_mm256_movemask_epi8(_mm256_cmpeq_epi8(high,target));
        return uint64_t(a)|(uint64_t(b)<<32);
    };
    return {matches(','),matches('"'),matches('\n')};
}
inline uint64_t below(unsigned n) {return (uint64_t(1)<<n)-1;}

template<size_t N> bool decode_masked(const char* p,CsvMask lo,CsvMask hi,unsigned length,
                                      std::array<Text,N>& out) {
    if (length<64) {
        lo.commas&=below(length);
        lo.quotes&=below(length);
    } else {
        hi.commas&=below(length-64);
        hi.quotes&=below(length-64);
    }
    unsigned low_count=std::popcount(lo.commas);
    if (low_count+std::popcount(hi.commas)!=N-1) return false;
    // Extract the quote flags immediately before and after every separator.
    // Matching flags and exactly two quotes per quoted field prove the dialect.
    uint64_t after=(lo.quotes>>1)|(hi.quotes<<63);
    unsigned opens=unsigned(lo.quotes&1) | unsigned(_pext_u64(after,lo.commas)<<1)
        | unsigned(_pext_u64(hi.quotes>>1,hi.commas)<<(low_count+1));
    uint64_t end_lo=lo.commas,end_hi=hi.commas;
    if (length<64) end_lo|=uint64_t(1)<<length;
    else end_hi|=uint64_t(1)<<(length-64);
    unsigned closes=unsigned(_pext_u64(lo.quotes<<1,end_lo))
        | unsigned(_pext_u64((hi.quotes<<1)|(lo.quotes>>63),end_hi)<<std::popcount(end_lo));
    if (opens!=closes || std::popcount(lo.quotes)+std::popcount(hi.quotes)!=2*std::popcount(opens))
        return false;
    unsigned first=0;
    for (unsigned i=0;i<N;++i) {
        unsigned end;
        if (end_lo) {end=std::countr_zero(end_lo); end_lo&=end_lo-1;}
        else {end=64+std::countr_zero(end_hi); end_hi&=end_hi-1;}
        unsigned trim=(opens>>i)&1;
        if (end-first<trim*2) return false;
        out[i]=Text(p+first+trim,end-first-trim*2);
        first=end+1;
    }
    return true;
}
#endif
template<size_t N,class F> void records(Text file,F visit) {
    std::array<Text,N> f;
    while (!file.empty()) {
#if defined(__AVX2__) && defined(__BMI2__)
        if (file.size()>=128) {
            const char* p=file.data();
            CsvMask lo=classify64(p),hi{};
            unsigned length=std::countr_zero(lo.newlines);
            if (length==64) {
                hi=classify64(p+64);
                length+=std::countr_zero(hi.newlines);
            }
            if (length<128) {
                if (decode_masked(p,lo,hi,length,f)) visit(file.substr(0,length),f);
                file.remove_prefix(length+1);
                continue;
            }
        }
#endif
        auto end=file.find('\n');
        auto line=file.substr(0,end);
        if (fields(line,f)) visit(line,f);
        if (end==Text::npos) break;
        file.remove_prefix(end+1);
    }
}
Text after_header(Text file) {
    auto end=file.find('\n');
    return end==Text::npos ? Text{} : file.substr(end+1);
}
template<size_t N,class F> void rows(Text file,F visit) {
    records<N>(after_header(file),visit);
}
int digits(Text s) {
    int n=0;
    for (char c:s) {
        if (c<'0' || c>'9') return -1;
        n=n*10+c-'0';
    }
    return n;
}
I64 number(Text s) {
    I64 value{};
    auto [end,ec]=std::from_chars(s.data(),s.data()+s.size(),value);
    return !s.empty() && s.front()>='0' && s.front()<='9' && ec==std::errc{} && end==s.data()+s.size()?value:bad;
}
// The window spans few UTC dates. Cache only validated calendar prefixes;
// a different prefix in the same slot always takes the full Gregorian path.
I64 midnight(Text s) {
    struct DateEntry { uint64_t prefix{}; uint16_t suffix{}; I64 base=bad; };
    static thread_local std::array<DateEntry,128> dates;
    unsigned slot=((s[6]&1)<<6)|((s[8]&3)<<4)|(s[9]&15);
    uint64_t prefix;
    uint16_t suffix;
    std::memcpy(&prefix,s.data(),8);
    std::memcpy(&suffix,s.data()+8,2);
    auto& entry=dates[slot];
    if (entry.prefix==prefix && entry.suffix==suffix) return entry.base;
    if (s[4]!='-' || s[7]!='-') return bad;
    int y=digits(s.substr(0,4)),m=digits(s.substr(5,2)),d=digits(s.substr(8,2));
    year_month_day date=year(y)/m/d;
    if (y<0 || !date.ok()) return bad;
    I64 base=I64(sys_days(date).time_since_epoch().count())*day_ms;
    entry={prefix,suffix,base};
    return base;
}
template<bool validate> int seconds_in_day(const char* p) {
    uint64_t bits;
    std::memcpy(&bits,p,8);
    if constexpr (std::endian::native!=std::endian::little) {
        int h=digits(Text(p,2)),m=digits(Text(p+3,2)),s=digits(Text(p+6,2));
        if (h<0 || h>23 || m<0 || m>59 || s<0 || s>59 || p[2]!=':' || p[5]!=':') return -1;
        return (h*60+m)*60+s;
    }
    constexpr uint64_t digit_bits=0xffff00ffff00ffffULL;
    if constexpr (validate) {
        constexpr uint64_t pattern=0x30303a30303a3030ULL;
        constexpr uint64_t high=0xf0f0fff0f0fff0f0ULL;
        if (((bits^pattern)&high)!=0) return -1;
        uint64_t low=bits & digit_bits & 0x0f0f0f0f0f0f0f0fULL;
        if ((low+0x0606060606060606ULL)&0x1010101010101010ULL) return -1;
    }
    constexpr uint64_t pairs=0x000f00000f00000fULL;
    uint64_t v=(bits&pairs)*10+((bits>>8)&pairs);
    unsigned h=v&255,m=(v>>24)&255,sec=v>>48;
    if constexpr (validate) if (h>=24 || m>=60 || sec>=60) return -1;
    return int((h*60+m)*60+sec);
}
I64 ingest_time(Text s) {
    if (s.size()!=20) return bad;
    return midnight(s)+seconds_in_day<false>(s.data()+11)*1000LL;
}
I64 epoch_millis(Text s) {
#ifdef __AVX2__
    if (s.size()==13) {
        uint64_t a,b;
        std::memcpy(&a,s.data(),8);
        std::memcpy(&b,s.data()+5,8);
        auto chars=_mm_set_epi64x(b,a);
        auto digit=_mm_sub_epi8(chars,_mm_set1_epi8('0'));
        if (!_mm_testz_si128(_mm_subs_epu8(digit,_mm_set1_epi8(9)),_mm_set1_epi8(-1))) return bad;
        auto ordered=_mm_shuffle_epi8(digit,_mm_setr_epi8(-1,-1,-1,0,1,2,3,4,8,9,10,11,12,13,14,15));
        auto pairs=_mm_maddubs_epi16(ordered,_mm_set1_epi16(0x010a));
        auto groups=_mm_madd_epi16(pairs,_mm_set1_epi32(0x00010064));
        I64 left=_mm_extract_epi32(groups,0)*10000+_mm_extract_epi32(groups,1);
        return left*100000000+_mm_extract_epi32(groups,2)*10000+_mm_extract_epi32(groups,3);
    }
#endif
    return number(s);
}
I64 timestamp(Text s) {
    if ((s.size()!=20 && s.size()!=25) || s[4]!='-') return epoch_millis(s);
    if (s[10]!='T') return bad;
    I64 base=midnight(s);
    int sec=seconds_in_day<true>(s.data()+11);
    if (base==bad || sec<0) return bad;
    if (s.size()==20) return s[19]=='Z' ? base+sec*1000LL : bad;
    int oh=digits(s.substr(20,2)),om=digits(s.substr(23,2));
    if ((s[19]!='+' && s[19]!='-') || s[22]!=':' || oh<0 || oh>=24 || om<0 || om>=60) return bad;
    int offset=(oh*60+om)*(s[19]=='+'?60:-60);
    return base+(sec-offset)*1000LL;
}
I64 manifest_time(Text doc,Text key) {
    auto p=doc.find(key);
    if (p==Text::npos) throw std::runtime_error("missing manifest key");
    p=doc.find('"',doc.find(':',p)+1);
    auto end=doc.find('"',p+1);
    auto t=timestamp(doc.substr(p+1,end-p-1));
    if (t==bad) throw std::runtime_error("invalid manifest timestamp");
    return t;
}
using Id = uint64_t;
constexpr Id missing_id=UINT64_MAX;

struct HashId {
    size_t operator()(Id key) const {
        key^=key>>32;
        key*=0xd6e8feb86659fd93ULL;
        return key^(key>>32);
    }
};

// Short prefixed IDs fit exactly in a word. All other byte strings use a
// disjoint dictionary range; equality is exact in both representations.
class Symbols {
    static constexpr Id external=Id(1)<<63;
    struct Pool {
        mutable std::mutex lock;
        Dict<Id> index;
        vector<Text> names;
    };
    char prefix;
    std::array<Pool,bins> pools;
public:
    explicit Symbols(char prefix):prefix(prefix) {}
    Id encode(Text name) {
        if constexpr (std::endian::native==std::endian::little) {
            if (!name.empty() && name.front()==prefix && name.size()<=9 && name.back()!=0) {
                Id key=0;
                if (name.size()==9) std::memcpy(&key,name.data()+1,8);
                else std::memcpy(&key,name.data()+1,name.size()-1);
                if (key<external) return key;
            }
        }
        size_t b=std::hash<Text>{}(name)%bins;
        auto& pool=pools[b];
        std::lock_guard guard(pool.lock);
        auto [it,fresh]=pool.index.try_emplace(name,external+2*(b+bins*pool.names.size()));
        if (fresh) pool.names.push_back(name);
        return it->second;
    }
    Text decode(Id key,std::array<char,9>& storage) const {
        if (key>=external) {
            auto index=(key-external)/2;
            auto& pool=pools[index%bins];
            std::lock_guard guard(pool.lock);
            return pool.names[index/bins];
        }
        storage[0]=prefix;
        std::memcpy(storage.data()+1,&key,8);
        return Text(storage.data(),1+(std::bit_width(key)+7)/8);
    }
};
// People already partitioned by hash use the remaining hash bits to probe.
template<unsigned skip_bits> class IdTable {
    struct Slot { Id key=missing_id,value=0; };
    vector<Slot> slots;
    size_t used=0;
    size_t locate(Id key) const {
        auto pos=(HashId{}(key)>>skip_bits) & (slots.size()-1);
        while (slots[pos].key!=missing_id && slots[pos].key!=key)
            pos=(pos+1)&(slots.size()-1);
        return pos;
    }
public:
    explicit IdTable(size_t estimate)
        :slots(std::bit_ceil(std::max<size_t>(16,estimate*3/2))) {}
    Id get_or_insert(Id key,Id initial) {
        auto pos=locate(key);
        if (slots[pos].key==key) return slots[pos].value;
        if ((used+1)*4>slots.size()*3) {
            auto old=std::move(slots);
            slots.assign(old.size()*2,{});
            for (auto entry:old) if (entry.key!=missing_id) slots[locate(entry.key)]=entry;
            pos=locate(key);
        }
        slots[pos]={key,initial};
        ++used;
        return initial;
    }
    Id find(Id key,Id fallback) const {
        const auto& slot=slots[locate(key)];
        return slot.key==missing_id?fallback:slot.value;
    }
    template<class F> void transform_values(F transform) {
        for (auto& entry:slots) if (entry.key!=missing_id) entry.value=transform(entry.value);
    }
};
// External symbol IDs are even. An odd value with the high bit set is
// a shard-local linked-person rank, so grouping needs no second lookup.
constexpr Id handle_tag=(Id(1)<<63)|1;
bool linked_rank(Id id) {return (id&handle_tag)==handle_tag;}
unsigned rank_of(Id id) {return unsigned((id^handle_tag)/(2*bins));}
size_t person_bin(Id id) {return (linked_rank(id)?id/2:HashId{}(id))%bins;}
struct Links {
    IdTable<0> table;
    vector<unsigned> parent, size;
    vector<Id> canonical;
    std::array<vector<Id>,bins> people;
    Symbols& symbols;
    Links(size_t estimate,Symbols& symbols):table(estimate),symbols(symbols) {}
    unsigned index(Text name) {
        Id id=symbols.encode(name);
        unsigned i=unsigned(table.get_or_insert(id,canonical.size()));
        if (i==canonical.size()) {
            canonical.push_back(id);
            parent.push_back(i);
            size.push_back(1);
        }
        return i;
    }
    unsigned root(unsigned i) {
        while (i!=parent[i]) i=parent[i]=parent[parent[i]];
        return i;
    }
    static Id lexical_word(Id key) {
#ifdef _MSC_VER
        return _byteswap_uint64(key);
#else
        return __builtin_bswap64(key);
#endif
    }
    void join(Text a,Text b) {
        unsigned i=root(index(a)),j=root(index(b));
        if (i==j) return;
        if (size[i]<size[j]) std::swap(i,j);
        parent[j]=i;
        size[i]+=size[j];
        if (canonical[i]<(Id(1)<<63) && canonical[j]<(Id(1)<<63)) {
            // Byte-swap so the first suffix byte is most significant.
            if (lexical_word(canonical[j])<lexical_word(canonical[i])) canonical[i]=canonical[j];
        } else {
            std::array<char,9> a_bytes,b_bytes;
            if (symbols.decode(canonical[j],b_bytes)<symbols.decode(canonical[i],a_bytes))
                canonical[i]=canonical[j];
        }
    }
    void finish() {
        vector<Id> handles(parent.size(),missing_id);
        table.transform_values([&](Id index) {
            unsigned r=root(unsigned(index));
            auto& handle=handles[r];
            if (handle==missing_id) {
                size_t b=HashId{}(canonical[r])%bins;
                handle=handle_tag|((people[b].size()*bins+b)*2);
                people[b].push_back(canonical[r]);
            }
            return handle;
        });
        vector<unsigned>().swap(parent);
        vector<unsigned>().swap(size);
        vector<Id>().swap(canonical);
    }
    Id person(Id id) const {
        return table.find(id,id);
    }
};
struct TimedEvent { I64 ts; int price,category; };
struct FullPayload {
    static TimedEvent pack(I64,TimedEvent e) {return e;}
    static TimedEvent unpack(I64,TimedEvent e) {return e;}
};
// Use the compact representation only when the entire window and catalog fit.
struct NarrowPayload {
    struct Data {uint32_t offset,purchase;};
    static Data pack(I64 start,TimedEvent e) {
        return {uint32_t(e.ts-start),(uint32_t(e.category)<<24)|uint32_t(e.price)};
    }
    static TimedEvent unpack(I64 start,Data e) {
        return {start+e.offset,int(e.purchase & 0xffffff),int(e.purchase>>24)};
    }
};
template<class Encoding> struct EventRecord {
    Id user=missing_id;
    decltype(Encoding::pack(0,TimedEvent{})) payload{};
};
template<class Event> struct DedupRecord {Id id; I64 ingest; Event event;};
struct Click {Id user; I64 ts; Text id; int campaign;};
template<class T> using Buckets = std::array<vector<T>,bins>;
size_t bucket(Id id) {return HashId{}(id)%bins;}
// Estimate rows from the local CSV density. Reserve once per partition;
// this affects allocation only, so long fields and skew can grow normally.
template<class T> void reserve_chunks(Buckets<T>& parts,Text chunk) {
    constexpr size_t sample_bytes=4096;
    if (chunk.size()<sample_bytes) return;
    auto sample=chunk.substr(0,sample_bytes);
    size_t lines=std::count(sample.begin(),sample.end(),'\n');
    auto average=std::min(chunk.size(),size_t(1024*1024))*lines/(sample_bytes*bins);
    if (average) for (auto& part:parts) part.reserve(average+average/4+4);
}
// CSV values borrow these bytes through the last Parquet write.
struct InputFile {
#ifdef __linux__
    std::shared_ptr<const char> mapping;
    size_t size{};
    void open(const fs::path& path) {
        size=fs::file_size(path);
        if (!size) return;
        int fd=::open(path.c_str(),O_RDONLY);
        if (fd<0) throw std::runtime_error("cannot open "+path.string());
        void* addr=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        close(fd);
        if (addr==MAP_FAILED) throw std::runtime_error("cannot map "+path.string());
        mapping={static_cast<char*>(addr),[bytes=size](const char* p){
            munmap(const_cast<char*>(p),bytes);
        }};
    }
    Text view() const { return size ? Text(mapping.get(),size) : Text{}; }
#else
    std::string bytes;
    void open(const fs::path& path) { bytes=read(path); }
    Text view() const { return bytes; }
#endif
};
struct Chunk {
    Text file;
    size_t offset;
    static constexpr size_t target=1024*1024;
    // Both neighbours compute the same newline boundary. Defer the first
    // page faults until a worker actually consumes this region.
    Text rows() const {
        auto first=file.find('\n',offset ? offset-1 : 0);
        if (first==Text::npos) return {};
        auto end=file.find('\n',std::min(file.size(),offset+target)-1);
        size_t stop=end==Text::npos ? file.size() : end+1;
        return file.substr(first+1,stop-first-1);
    }
};
struct Input {
    vector<InputFile> files;
    vector<Chunk> chunks;
    explicit Input(const fs::path& folder) {
        vector<fs::path> paths;
        for (auto& p:fs::directory_iterator(folder))
            if (p.path().extension()==".csv") paths.push_back(p.path());
        files.resize(paths.size());
        parallel(paths.size(),[&](size_t i){files[i].open(paths[i]);});
        for (auto& file:files) {
            auto view=file.view();
            for (size_t offset=0;offset<view.size();offset+=Chunk::target)
                chunks.push_back({view,offset});
        }
    }
};
struct Counts {
    I64 sessions{}, purchasing{}, revenue{}, assists{};
};
struct Results {
    vector<UserRow> users;
    std::unordered_map<int,vector<Counts>> days;
};
struct People {
    IdTable<std::countr_zero(unsigned(bins))> ids;
    vector<Id> names;
    People(size_t estimate,vector<Id> known):ids(estimate),names(std::move(known)) {}
    unsigned intern(Id name) {
        if (linked_rank(name)) return rank_of(name);
        unsigned rank=unsigned(ids.get_or_insert(name,names.size()));
        if (rank==names.size()) names.push_back(name);
        return rank;
    }
    static constexpr unsigned missing=std::numeric_limits<unsigned>::max();
    unsigned find(Id name) const { return linked_rank(name)?rank_of(name):unsigned(ids.find(name,missing)); }
};
struct TimedClick { I64 ts; Text id; int campaign; };
template<class T> struct Timeline {
    vector<T> rows;
    vector<size_t> offsets;
};
// Counting-sort by person so only the timestamps inside each person's slice
// need comparison sorting. Click-only people have no attributable sessions.
template<class T,class Row,class Identify,class Project>
Timeline<T> group_by_person(vector<Buckets<Row>>& sources,size_t bin,size_t known_people,
                            Identify identify,Project project) {
    vector<unsigned> owners;
    size_t total=0;
    for (auto& source:sources) total+=source[bin].size();
    owners.reserve(total);
    vector<size_t> counts(known_people);
    for (auto& source:sources) for (const auto& row:source[bin]) {
        unsigned id=identify(row.user);
        owners.push_back(id);
        if (id==People::missing) continue;
        if (id==counts.size()) counts.push_back(0);
        ++counts[id];
    }
    Timeline<T> result;
    result.offsets.resize(counts.size()+1);
    std::partial_sum(counts.begin(),counts.end(),result.offsets.begin()+1);
    result.rows.resize(result.offsets.back());
    auto next=result.offsets;
    size_t index=0;
    for (auto& source:sources) {
        for (const auto& row:source[bin]) {
            unsigned id=owners[index++];
            if (id!=People::missing) result.rows[next[id]++]=project(row);
        }
        vector<Row>().swap(source[bin]);
    }
    return result;
}
void aggregate(Timeline<TimedEvent>& event_groups, Timeline<TimedClick>& click_groups,
               const vector<Text>& people, const vector<Text>& category_names,
               size_t campaign_count, Results& out) {
    auto& events=event_groups.rows;
    auto& clicks=click_groups.rows;
    vector<int> counts(campaign_count), position(campaign_count), active;
    vector<I64> categories(category_names.size());
    vector<int> purchased;
    for (size_t person=0;person<people.size();++person) {
        Text user=people[person];
        size_t e=event_groups.offsets[person],event_end=event_groups.offsets[person+1];
        if (e==event_end) continue;
        size_t c=click_groups.offsets[person],click_end=click_groups.offsets[person+1];
        std::sort(events.begin()+e,events.begin()+event_end,[](auto& a,auto& b){
            return a.ts<b.ts;
        });
        // The last eligible click must win the lexicographically smallest ID tie.
        std::sort(clicks.begin()+c,clicks.begin()+click_end,[](auto& a,auto& b){
            return a.ts==b.ts ? a.id>b.id : a.ts<b.ts;
        });
        size_t left=c;
        for (int id:active) counts[id]=0;
        active.clear();
        I64 best=0;
        while (e<event_end) {
            I64 start=events[e].ts,prev=start, revenue=0;
            int day=int(floor<days>(milliseconds(start)).count());
            if (out.users.empty() || out.users.back().user!=user || out.users.back().day!=day) {
                out.users.push_back({user,day});
                for (int id:purchased) categories[id]=0;
                purchased.clear();
                best=0;
            }
            auto& u=out.users.back();
            ++u.sessions;
            do {
                auto& event=events[e++];
                prev=event.ts;
                ++u.events;
                revenue+=event.price;
                if (event.price) {
                    if (!categories[event.category]) purchased.push_back(event.category);
                    auto total=(categories[event.category]+=event.price);
                    if (total>best || (total==best && category_names[event.category]<u.top)) {
                        best=total;
                        u.top=category_names[event.category];
                    }
                }
            } while (e<event_end && events[e].ts-prev<=1800000);
            u.revenue+=revenue;
            while (c<click_end && clicks[c].ts<=start) {
                int id=clicks[c++].campaign;
                if (!counts[id]++) {
                    position[id]=int(active.size());
                    active.push_back(id);
                }
            }
            while (left<c && clicks[left].ts<start-week) {
                int id=clicks[left++].campaign;
                if (!--counts[id]) {
                    auto slot=position[id];
                    position[active.back()]=slot;
                    active[slot]=active.back();
                    active.pop_back();
                }
            }
            auto& daily=out.days[day];
            if (daily.empty()) daily.resize(campaign_count);
            int credit=left<c?clicks[c-1].campaign:0;
            auto& credited=daily[credit];
            ++credited.sessions;
            credited.purchasing+=revenue>0;
            credited.revenue+=revenue;
            for (int id:active) ++daily[id].assists;
        }
    }
}
// Dimension IDs often have a short decimal suffix. Build the direct index
// from the actual catalog and keep exact string lookup for every other key.
struct Dimension {
    Text prefix;
    vector<int> direct;
    Dict<int> other;
    explicit Dimension(Text prefix):prefix(prefix) {}
    int suffix(Text key) const {
        if (!key.starts_with(prefix)) return -1;
        auto s=key.substr(prefix.size());
        if (s.empty() || s.size()>6 || (s.size()>1 && s[0]=='0')) return -1;
        return digits(s);
    }
    void add(Text key,int value) {
        int n=suffix(key);
        if (n<0) {other.emplace(key,value);return;}
        if (size_t(n)>=direct.size()) direct.resize(n+1,-1);
        direct[n]=value;
    }
    int find(Text key) const {
        int n=suffix(key);
        if (n>=0) return size_t(n)<direct.size()?direct[n]:-1;
        auto it=other.find(key);
        return it==other.end()?-1:it->second;
    }
};
void run(fs::path data,fs::path output) {
    auto last=steady_clock::now();
    const bool profile=std::getenv("SHOP_PROFILE")!=nullptr;
    auto mark=[&](const char* phase) {
        if (!profile) return;
        auto now=steady_clock::now();
        std::cerr << phase << " " << duration<double>(now-last).count() << " s\n";
        last=now;
    };
    auto manifest=read(data/"manifest.json");
    I64 start=manifest_time(manifest,"window_start"),end=manifest_time(manifest,"window_end");
    auto products=read(data/"products.csv"),campaigns=read(data/"campaigns.csv");
    Dimension campaign_ids("c_"),product_categories("p");
    vector<std::pair<Text,Text>> campaign_names{{"organic","organic"}};
    Dict<int> category_ids{{"unknown",0}};
    vector<Text> category_names{"unknown"};
    rows<2>(products,[&](Text,auto f){
        auto [it,fresh]=category_ids.try_emplace(f[1],int(category_names.size()));
        if (fresh) category_names.push_back(f[1]);
        product_categories.add(f[0],it->second);
    });
    rows<2>(campaigns,[&](Text,auto f){
        campaign_ids.add(f[0],int(campaign_names.size()));
        campaign_names.emplace_back(f[0],f[1]);
    });
    Input links(data/"links");
    size_t link_bytes=0;
    for (auto& file:links.files) link_bytes+=file.view().size();
    Symbols users('u'),event_ids('e');
    Links identity(link_bytes/20,users);
    Input events(data/"events"),clicks(data/"clicks");
    mark("map inputs and load dimensions");
    auto process=[&]<class Encoding>() {
        using Event=EventRecord<Encoding>;
        using Record=DedupRecord<Event>;
        vector<Buckets<Record>> raw(events.chunks.size());
        // Parsing needs the raw user ID only. Build the identity graph on one
        // of the same eight workers, then resolve surviving IDs after dedup.
        parallel(events.chunks.size()+1,[&](size_t task){
            if (task==0) {
                for (auto chunk:links.chunks) records<2>(chunk.rows(),[&](Text,auto f){
                    if (!f[0].empty() && !f[1].empty()) identity.join(f[0],f[1]);
                });
                identity.finish();
                return;
            }
            size_t i=task-1;
            auto chunk=events.chunks[i].rows();
            reserve_chunks(raw[i],chunk);
            records<7>(chunk,[&](Text,auto f){
                if (f[0].empty()) return;
                I64 ingest=ingest_time(f[3]);
                // Keep even an invalid earliest copy in the deduplication contest.
                Id id=event_ids.encode(f[0]);
                auto& r=raw[i][bucket(id)].emplace_back(Record{id,ingest,{}});
                if (f[1].empty() || f[5].empty()) return;
                I64 t=timestamp(f[2]);
                if (t<start || t>=end || ingest>t+day_ms) return;
                int price=0,category=0;
                if (f[4]=="purchase") {
                    auto value=number(f[6]);
                    if (value<=0 || value>10000000) return;
                    price=int(value);
                    category=std::max(0,product_categories.find(f[5]));
                } else if (f[4]!="view" && f[4]!="cart") return;
                r.event={users.encode(f[1]),Encoding::pack(start,{t,price,category})};
            });
        });
        mark("resolve links and parse event CSV");
        vector<Buckets<Event>> parsed(bins);
        parallel(bins,[&](size_t b){
            size_t n=0;
            for (auto& parts:raw) n+=parts[b].size();
            struct Flags {uint64_t seen=0,repeated=0;};
            size_t bits=std::bit_ceil(std::max<size_t>(64,4*n));
            vector<Flags> flags(bits/64);
            size_t revisits=0;
            for (auto& parts:raw) for (const auto& row:parts[b]) {
                size_t h=(HashId{}(row.id)/bins)&(bits-1);
                uint64_t bit=uint64_t(1)<<(h%64);
                auto& word=flags[h/64];
                auto seen=word.seen&bit;
                word.repeated|=seen;
                word.seen|=bit;
                revisits+=seen!=0;
            }
            for (auto& part:parsed[b]) part.reserve(n/bins+n/(bins*8));
            auto emit=[&](Event ev) {
                if (ev.user==missing_id) return;
                ev.user=identity.person(ev.user);
                parsed[b][person_bin(ev.user)].push_back(ev);
            };
            // A singleton slot proves uniqueness. For a slot containing k>=2
            // rows, k <= 2*(k-1), so two times revisits bounds the delayed rows.
            size_t capacity=std::bit_ceil(3*revisits+1);
            vector<unsigned> table(capacity,UINT32_MAX);
            vector<Record> delayed;
            delayed.reserve(2*revisits);
            for (auto& parts:raw) for (const auto& row:parts[b]) {
                size_t hash=HashId{}(row.id)/bins;
                size_t h=hash&(bits-1);
                if (!(flags[h/64].repeated & (uint64_t(1)<<(h%64)))) {
                    emit(row.event);
                    continue;
                }
                size_t pos=hash&(capacity-1);
                while (table[pos]!=UINT32_MAX && delayed[table[pos]].id!=row.id)
                    pos=(pos+1)&(capacity-1);
                if (table[pos]==UINT32_MAX) {
                    table[pos]=unsigned(delayed.size());
                    delayed.push_back(row);
                } else if (row.ingest<delayed[table[pos]].ingest) {
                    delayed[table[pos]]=row;
                }
            }
            for (const auto& row:delayed) emit(row.event);
            for (auto& parts:raw) vector<Record>().swap(parts[b]);
        });
        mark("deduplicate and partition by person");
        vector<Buckets<Click>> ads(clicks.chunks.size());
        parallel(clicks.chunks.size(),[&](size_t i){
            auto chunk=clicks.chunks[i].rows();
            reserve_chunks(ads[i],chunk);
            records<4>(chunk,[&](Text,auto f){
                auto campaign=campaign_ids.find(f[2]);
                if (f[0].empty() || f[1].empty() || campaign<0) return;
                auto t=timestamp(f[3]);
                if (t<start-week || t>=end) return;
                auto user=identity.person(users.encode(f[1]));
                ads[i][person_bin(user)].push_back({user,t,f[0],campaign});
            });
        });
        mark("prepare clicks");
        fs::create_directories(output/"user_daily");
        fs::create_directories(output/"campaign_daily");
        vector<Results> results(bins);
        parallel(bins,[&](size_t b){
            size_t event_count=0;
            for (auto& part:parsed) event_count+=part[b].size();
            People people(event_count/64,std::move(identity.people[b]));
            auto ev=group_by_person<TimedEvent>(parsed,b,people.names.size(),[&](Id u){return people.intern(u);},
                [start](const Event& e){return Encoding::unpack(start,e.payload);});
            auto ad=group_by_person<TimedClick>(ads,b,people.names.size(),[&](Id u){return people.find(u);},
                [](const Click& c){return TimedClick{c.ts,c.id,c.campaign};});
            vector<std::array<char,9>> storage(people.names.size());
            vector<Text> names;
            names.reserve(storage.size());
            for (size_t i=0;i<storage.size();++i) names.push_back(users.decode(people.names[i],storage[i]));
            aggregate(ev,ad,names,category_names,campaign_names.size(),results[b]);
            auto& rows=results[b].users;
            if (!rows.empty() || b==0) save(output/"user_daily"/(std::to_string(b)+".parquet"),rows);
            vector<UserRow>().swap(rows);
        });
        mark("sort, aggregate and write user tables");
        std::unordered_map<int,vector<Counts>> merged;
        for (auto& part:results) for (auto& [day,rows]:part.days) {
            auto& daily=merged[day];
            if (daily.empty()) daily.resize(campaign_names.size());
            for (size_t i=0;i<rows.size();++i) {
                auto& m=daily[i];
                m.sessions+=rows[i].sessions;
                m.purchasing+=rows[i].purchasing;
                m.revenue+=rows[i].revenue;
                m.assists+=rows[i].assists;
            }
        }
        vector<CampaignRow> totals;
        for (auto& [day,rows]:merged) for (size_t i=0;i<rows.size();++i) {
            auto& r=rows[i];
            if (!(r.sessions || r.assists)) continue;
            auto [campaign,channel]=campaign_names[i];
            totals.push_back({campaign,channel,day,r.sessions,r.purchasing,r.revenue,r.assists});
        }
        save(output/"campaign_daily"/"0.parquet",totals);
        mark("merge and write campaign table");
    };
    if (end>start && uint64_t(end-start)<=UINT32_MAX && category_names.size()<=256)
        process.template operator()<NarrowPayload>();
    else process.template operator()<FullPayload>();
#ifdef __linux__
    if (profile) {
        rusage usage{};
        if (getrusage(RUSAGE_SELF,&usage)==0)
            std::cerr << "peak RSS " << usage.ru_maxrss/1024 << " MiB\n";
    }
#endif

}
int main(int argc,char** argv) {
    try {
        if (auto count=std::getenv("NCPU")) threads=std::clamp(std::stoi(count),1,64);
        auto path=[&](int i,const char* env,const char* fallback) {
            if (i<argc) return static_cast<const char*>(argv[i]);
            return std::getenv(env)?std::getenv(env):fallback;
        };
        run(path(1,"DATA_DIR","/data"),path(2,"OUTPUT_DIR","/output"));
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
