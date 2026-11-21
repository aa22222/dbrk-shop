#pragma once
// Shared CSV dialect, timestamp decoder and union-find from the scored C++ reader.
#include <filesystem>
#include <vector>
#include <memory>
#include <string_view>
#include <cstdlib>
#include <cstdint>
using Text=std::string_view;
using I64=int64_t;
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
template<size_t N,class F> void records(Text file,F visit) {
    if (file.empty()) return;
#ifdef __AVX2__
    // Locate separators and count quotes in 32-byte blocks. A row is valid
    // only when every quote belongs to a checked pair around a whole field.
    const char* line=file.data();
    const char* field=line;
    std::array<Text,N> values;
    size_t count=0,quotes=0,paired=0;
    bool valid=true;
    auto finish_field=[&](const char* end) {
        Text value(field,size_t(end-field));
        if (value.starts_with('"')) {
            if (value.size()<2 || value.back()!='"') valid=false;
            else {
                value=value.substr(1,value.size()-2);
                paired+=2;
            }
        }
        if (count<N) values[count]=value;
        ++count;
    };
    auto finish_line=[&](const char* end) {
        if (valid && count==N && quotes==paired) visit(Text(line,size_t(end-line)),values);
        valid=true;
        count=quotes=paired=0;
    };
    for (size_t off=0;off<file.size();off+=32) {
        const char* block=file.data()+off;
        size_t length=std::min<size_t>(32,file.size()-off);
        alignas(32) char tail[32]{};
        const char* load=block;
        if (length<32) {
            std::memcpy(tail,block,length);
            load=tail;
        }
        auto bytes=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(load));
        auto comma=_mm256_cmpeq_epi8(bytes,_mm256_set1_epi8(','));
        auto newline=_mm256_cmpeq_epi8(bytes,_mm256_set1_epi8('\n'));
        auto quote=_mm256_cmpeq_epi8(bytes,_mm256_set1_epi8('"'));
        uint32_t separators=_mm256_movemask_epi8(_mm256_or_si256(comma,newline));
        uint32_t quote_bits=_mm256_movemask_epi8(quote);
        while (separators) {
            unsigned n=std::countr_zero(separators);
            separators&=separators-1;
            const char* end=block+n;
            finish_field(end);
            if (*end=='\n') {
                uint32_t before=(uint32_t(1)<<n)-1;
                quotes+=std::popcount(quote_bits & before);
                quote_bits&=~before;
                finish_line(end);
                line=end+1;
            }
            field=end+1;
        }
        quotes+=std::popcount(quote_bits);
    }
    if (line!=file.data()+file.size()) {
        auto end=file.data()+file.size();
        finish_field(end);
        finish_line(end);
    }
#else
    while (!file.empty()) {
        auto end=file.find('\n');
        auto line=file.substr(0,end);
        std::array<Text,N> f;
        if (fields(line,f)) visit(line,f);
        if (end==Text::npos) break;
        file.remove_prefix(end+1);
    }
#endif
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
int two_digits(const char* p) {
    return (p[0]-'0')*10+p[1]-'0';
}
I64 ingest_time(Text s) {
    // ingest_ts is guaranteed valid F1.
    if (s.size()!=20) return bad;
    int h=two_digits(s.data()+11),m=two_digits(s.data()+14),sec=two_digits(s.data()+17);
    return midnight(s)+((h*60+m)*60+sec)*1000LL;
}
I64 timestamp(Text s) {
    if ((s.size()!=20 && s.size()!=25) || s[4]!='-') return number(s);
    if (s[10]!='T' || s[13]!=':' || s[16]!=':') return bad;
    auto base=midnight(s);
    if (base==bad) return bad;
    int h=digits(s.substr(11,2)), mi=digits(s.substr(14,2)), sec=digits(s.substr(17,2));
    if (h<0 || h>=24 || mi<0 || mi>=60 || sec<0 || sec>=60) return bad;
    int offset=0;
    if (s.size()==20) {
        if (s[19]!='Z') return bad;
    } else {
        int oh=digits(s.substr(20,2)), om=digits(s.substr(23,2));
        if ((s[19]!='+' && s[19]!='-') || s[22]!=':' || oh<0 || oh>=24 || om<0 || om>=60) return bad;
        offset=(oh*60+om)*(s[19]=='+'?1:-1);
    }
    return base+((h*60+mi-offset)*60+sec)*1000LL;
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
        auto [it,fresh]=pool.index.try_emplace(name,external+b+bins*pool.names.size());
        if (fresh) pool.names.push_back(name);
        return it->second;
    }
    Text decode(Id key,std::array<char,9>& storage) const {
        if (key>=external) {
            auto index=key-external;
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
struct Links {
    IdTable<0> table;
    vector<unsigned> parent, size;
    vector<Id> canonical;
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
        table.transform_values([&](Id index){return canonical[root(unsigned(index))];});
        vector<unsigned>().swap(parent);
        vector<unsigned>().swap(size);
        vector<Id>().swap(canonical);
    }
    Id person(Id id) const {
        return table.find(id,id);
    }
};
struct Event {Id user=missing_id; I64 ts; int price, category;};
struct Record {Id id; I64 ingest; Event event;};
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
