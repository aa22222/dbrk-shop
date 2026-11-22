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
#include <map>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace std::chrono;
using std::vector;
template<class V> using Dict = std::map<Text,V>;
constexpr int bins = 64;
constexpr size_t chunk_bytes = 8 * 1024 * 1024;
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

// Use digit bits as a direct array address. All equality tests use the
// original word, including the byte bits omitted from this address.
Id digit_bits(Id word) {
    word=std::rotr(word,16);
#ifdef __BMI2__
    return _pext_u64(word,0x1f1f1f1f1f1f1f1fULL);
#else
    Id address=0;
    for (unsigned i=0;i<8;++i) address|=((word>>(8*i))&31)<<(5*i);
    return address;
#endif
}

Id person_digits(Id id) {
    return id>>63 ? id/2 : digit_bits(id);
}
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
        size_t b=static_cast<unsigned char>(name.back())%bins;
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
// Each radix bucket is one contiguous slice of exact, sorted IDs.
struct FlatIds {
    vector<Id> keys;
    vector<unsigned> offsets;
    unsigned skip;
    explicit FlatIds(unsigned skip=0):skip(skip) {}
    size_t bucket(Id key) const { return (person_digits(key)>>skip)&(offsets.size()-2); }
    void build(vector<Id> input) {
        size_t buckets=std::bit_ceil(std::max<size_t>(1,input.size()/2));
        offsets.assign(buckets+1,0);
        for (Id key:input) ++offsets[bucket(key)+1];
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        auto next=offsets;
        vector<Id> grouped(input.size());
        for (Id key:input) grouped[next[bucket(key)]++]=key;
        keys=std::move(input);
        size_t used=0;
        for (size_t b=0;b<buckets;++b) {
            auto first=grouped.begin()+offsets[b],last=grouped.begin()+offsets[b+1];
            std::sort(first,last);
            last=std::unique(first,last);
            offsets[b]=unsigned(used);
            std::copy(first,last,keys.begin()+used);
            used+=size_t(last-first);
        }
        offsets.back()=unsigned(used);
        keys.resize(used);
    }
    unsigned find(Id key) const {
        size_t b=bucket(key);
        auto first=keys.begin()+offsets[b],last=keys.begin()+offsets[b+1];
        if (last-first<=8) {
            for (auto it=first;it!=last;++it) if (*it==key) return unsigned(it-keys.begin());
            return UINT32_MAX;
        }
        auto it=std::lower_bound(first,last,key);
        return it!=last && *it==key ? unsigned(it-keys.begin()) : UINT32_MAX;
    }
};
// External symbol IDs are even. An odd value with the high bit set is
// a shard-local linked-person rank, so grouping needs no second lookup.
constexpr Id handle_tag=(Id(1)<<63)|1;
bool linked_rank(Id id) {return (id&handle_tag)==handle_tag;}
unsigned rank_of(Id id) {return unsigned((id^handle_tag)/(2*bins));}
size_t person_bin(Id id) {return (linked_rank(id)?id/2:person_digits(id))%bins;}
struct Links {
    FlatIds aliases;
    vector<std::pair<Id,Id>> edges;
    vector<unsigned> parent,size;
    vector<Id> canonical;
    struct Alias {Id key,person;};
    vector<Alias> resolved;
    std::array<vector<Id>,bins> people;
    Symbols& symbols;
    Links(size_t estimate,Symbols& symbols):symbols(symbols) {edges.reserve(estimate);}
    void join(Text a,Text b) {edges.emplace_back(symbols.encode(a),symbols.encode(b));}
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
    void unite(Id a,Id b) {
        unsigned i=root(aliases.find(a)),j=root(aliases.find(b));
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
        vector<Id> endpoints;
        endpoints.reserve(2*edges.size());
        for (auto [a,b]:edges) {endpoints.push_back(a); endpoints.push_back(b);}
        aliases.build(std::move(endpoints));
        canonical=aliases.keys;
        parent.resize(canonical.size());
        std::iota(parent.begin(),parent.end(),0u);
        size.assign(parent.size(),1);
        for (auto [a,b]:edges) unite(a,b);
        decltype(edges)().swap(edges);
        vector<Id> handles(parent.size(),missing_id);
        resolved.resize(parent.size());
        for (size_t index=0;index<parent.size();++index) {
            unsigned r=root(unsigned(index));
            auto& handle=handles[r];
            if (handle==missing_id) {
                size_t b=person_digits(canonical[r])%bins;
                handle=handle_tag|((people[b].size()*bins+b)*2);
                people[b].push_back(canonical[r]);
            }
            resolved[index]={aliases.keys[index],handle};
        }
        vector<unsigned>().swap(parent);
        vector<unsigned>().swap(size);
        vector<Id>().swap(canonical);
        vector<Id>().swap(aliases.keys);
    }
    Id person(Id id) const {
        size_t b=aliases.bucket(id);
        auto begin=resolved.begin()+aliases.offsets[b],end=resolved.begin()+aliases.offsets[b+1];
        if (end-begin<=8) {
            for (auto it=begin;it!=end;++it) if (it->key==id) return it->person;
            return id;
        }
        auto it=std::lower_bound(begin,end,id,[](const Alias& a,Id key){return a.key<key;});
        return it!=end && it->key==id ? it->person : id;
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
template<class Key, class Event> struct DedupRecord {Key id; I64 ingest; Event event;};
struct Click {Id user; I64 ts; Text id; int campaign;};
template<class T> using Buckets = std::array<vector<T>,bins>;
// Generated IDs are canonical base-36 strings. Other spellings, including
// leading zeroes, are grouped by their original bytes.
Id event_number(Text text) {
    if (text.size()<2 || text.size()>9 || text[0]!='e' ||
        (text.size()>2 && text[1]=='0')) return missing_id;
    static constexpr auto digits=[] {
        std::array<unsigned char,256> table{};
        table.fill(255);
        for (int i=0;i<10;++i) table['0'+i]=i;
        for (int i=0;i<26;++i) table['a'+i]=10+i;
        return table;
    }();
    // An eight-digit suffix can be read as four base-36 pairs.
    if constexpr (std::endian::native==std::endian::little) {
        if (text.size()==9) {
            static constexpr auto pairs=[] {
                std::array<uint16_t,65536> result{};
                result.fill(UINT16_MAX);
                for (unsigned a='0';a<='z';++a) for (unsigned b='0';b<='z';++b)
                    if (digits[a]!=255 && digits[b]!=255)
                        result[a+256*b]=uint16_t(36*digits[a]+digits[b]);
                return result;
            }();
            std::array<uint16_t,4> words;
            std::memcpy(words.data(),text.data()+1,8);
            Id a=pairs[words[0]],b=pairs[words[1]],c=pairs[words[2]],d=pairs[words[3]];
            if ((a|b|c|d)&0x8000) return missing_id;
            return (a*1296+b)*(1296*1296)+(c*1296+d);
        }
    }
    Id number=0;
    for (unsigned char c:text.substr(1)) {
        auto digit=digits[c];
        if (digit==255) return missing_id;
        number=number*36+digit;
    }
    return number;
}

template<class Record, class Emit>
void emit_winners(vector<Record>& rows,Emit emit) {
    for (size_t i=0;i<rows.size();) {
        auto* winner=&rows[i++];
        while (i<rows.size() && rows[i].id==winner->id) {
            if (rows[i].ingest<winner->ingest) winner=&rows[i];
            ++i;
        }
        emit(winner->event);
    }
}

// Estimate rows from the local CSV density. Reserve once per partition;
// this affects allocation only, so long fields and skew can grow normally.
template<class T> void reserve_chunks(Buckets<T>& parts,Text chunk) {
    constexpr size_t sample_bytes=4096;
    if (chunk.size()<sample_bytes) return;
    auto sample=chunk.substr(0,sample_bytes);
    size_t lines=std::count(sample.begin(),sample.end(),'\n');
    auto average=std::min(chunk.size(),chunk_bytes)*lines/(sample_bytes*bins);
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
    static constexpr size_t target=chunk_bytes;
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
struct DayIndex {
    int first;
    vector<unsigned> index;
    DayIndex(I64 start,I64 end):first(int(floor<days>(milliseconds(start)).count())) {
        if (end>start) index.resize(size_t(floor<days>(milliseconds(end-1)).count()-first+1));
    }
    using Rows=vector<std::pair<int,vector<Counts>>>;
    vector<Counts>& get(Rows& rows,int day,size_t campaigns) {
        auto& pos=index[size_t(day-first)];
        if (!pos) {
            rows.emplace_back(day,vector<Counts>(campaigns));
            pos=unsigned(rows.size());
        }
        return rows[pos-1].second;
    }
};
struct Results {
    vector<UserRow> users;
    DayIndex::Rows days;
};
struct People {
    static constexpr unsigned missing=UINT32_MAX;
    FlatIds unlinked{std::countr_zero(unsigned(bins))};
    vector<Id> names;
    unsigned known;
    People(vector<Id> linked,vector<Id> others):names(std::move(linked)),known(unsigned(names.size())) {
        unlinked.build(std::move(others));
        names.insert(names.end(),unlinked.keys.begin(),unlinked.keys.end());
    }
    unsigned find(Id name) const {
        if (linked_rank(name)) return rank_of(name);
        unsigned i=unlinked.find(name);
        return i==missing ? missing : known+i;
    }
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
template<class Encoding,class Data>
void aggregate(I64 window_start,Timeline<Data>& event_groups, Timeline<TimedClick>& click_groups,
               const vector<Text>& people, const vector<Text>& category_names,
               size_t campaign_count, DayIndex day_index, Results& out) {
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
        std::sort(events.begin()+e,events.begin()+event_end,[window_start](auto& a,auto& b){
            return Encoding::unpack(window_start,a).ts<Encoding::unpack(window_start,b).ts;
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
            I64 start=Encoding::unpack(window_start,events[e]).ts,prev=start, revenue=0;
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
                auto event=Encoding::unpack(window_start,events[e++]);
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
            } while (e<event_end && Encoding::unpack(window_start,events[e]).ts-prev<=1800000);
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
            auto& daily=day_index.get(out.days,day,campaign_count);
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
    Symbols users('u');
    Links identity(link_bytes/20,users);
    Input events(data/"events"),clicks(data/"clicks");
    mark("map inputs and load dimensions");
    auto process=[&]<class Encoding>() {
        using Event=EventRecord<Encoding>;
        using Record=DedupRecord<Id,Event>;
        using Other=DedupRecord<Text,Event>;
        vector<vector<Other>> unusual(events.chunks.size());
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
                Id id=event_number(f[0]);
                Event& event=id==missing_id
                    ? unusual[i].emplace_back(Other{f[0],ingest,{}}).event
                    : raw[i][id%bins].emplace_back(Record{id,ingest,{}}).event;
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
                event={users.encode(f[1]),Encoding::pack(start,{t,price,category})};
            });
        });
        mark("resolve links and parse event CSV");
        vector<Buckets<Event>> parsed(bins+1);
        auto deduplicate=[&](size_t b){
            auto emit=[&](Event event) {
                if (event.user==missing_id) return;
                event.user=identity.person(event.user);
                parsed[b][person_bin(event.user)].push_back(event);
            };
            if (b==bins) {
                vector<Other> rows;
                for (auto& part:unusual) rows.insert(rows.end(),part.begin(),part.end());
                std::sort(rows.begin(),rows.end(),[](auto& a,auto& b){return a.id<b.id;});
                emit_winners(rows,emit);
                return;
            }
            size_t n=0;
            for (const auto& parts:raw) n+=parts[b].size();
            for (auto& part:parsed[b]) part.reserve(n/bins);
            if (!n) return;
            // A one-byte counter per low-digit address is sufficient. Once
            // the count reaches two, send those full IDs to the exact sort.
            const size_t mask=std::bit_ceil(8*n)-1;
            vector<unsigned char> counts(mask+1);
            for (const auto& parts:raw) for (const auto& row:parts[b]) {
                auto& count=counts[(row.id/bins)&mask];
                if (count<2) ++count;
            }
            vector<Record> repeated;
            for (auto& parts:raw) {
                for (const auto& row:parts[b]) {
                    if (counts[(row.id/bins)&mask]==1) emit(row.event);
                    else repeated.push_back(row);
                }
                vector<Record>().swap(parts[b]);
            }
            std::sort(repeated.begin(),repeated.end(),[](auto& a,auto& b){return a.id<b.id;});
            emit_winners(repeated,emit);
        };
        vector<Buckets<Click>> ads(clicks.chunks.size());
        auto parse_clicks=[&](size_t i){
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
        };
        parallel(parsed.size()+ads.size(),[&](size_t task){
            if (task<ads.size()) parse_clicks(task);
            else deduplicate(task-ads.size());
        });
        mark("deduplicate events, prepare clicks and partition by person");
        fs::create_directories(output/"user_daily");
        fs::create_directories(output/"campaign_daily");
        vector<Results> results(bins);
        parallel(bins,[&](size_t b){
            vector<Id> unlinked;
            for (auto& part:parsed) for (const auto& event:part[b])
                if (!linked_rank(event.user)) unlinked.push_back(event.user);
            People people(std::move(identity.people[b]),std::move(unlinked));
            auto ev=group_by_person<decltype(Event{}.payload)>(parsed,b,people.names.size(),[&](Id u){return people.find(u);},
                [](const Event& e){return e.payload;});
            auto ad=group_by_person<TimedClick>(ads,b,people.names.size(),[&](Id u){return people.find(u);},
                [](const Click& c){return TimedClick{c.ts,c.id,c.campaign};});
            vector<std::array<char,9>> storage(people.names.size());
            vector<Text> names;
            names.reserve(storage.size());
            for (size_t i=0;i<storage.size();++i) names.push_back(users.decode(people.names[i],storage[i]));
            aggregate<Encoding>(start,ev,ad,names,category_names,campaign_names.size(),DayIndex(start,end),results[b]);
            auto& rows=results[b].users;
            if (!rows.empty() || b==0) save(output/"user_daily"/(std::to_string(b)+".parquet"),rows);
            vector<UserRow>().swap(rows);
        });
        mark("sort, aggregate and write user tables");
        DayIndex day_index(start,end);
        DayIndex::Rows merged;
        for (auto& part:results) for (auto& [day,rows]:part.days) {
            auto& daily=day_index.get(merged,day,campaign_names.size());
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
