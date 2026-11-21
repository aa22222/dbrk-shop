#include "reader_common.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;

struct EventColumns {
    vector<Id> id,user;
    vector<I64> ingest,ts;
    vector<int> price,category;
};
struct ClickColumns {
    vector<Id> user;
    vector<I64> ts;
    vector<int> campaign;
    vector<Text> id;
};
template<class T> py::array_t<T> array(vector<T>&& values) {
    auto owned=std::make_unique<vector<T>>(std::move(values));
    size_t count=owned->size();
    T* data=owned->data();
    py::capsule holder(owned.release(),[](void* v){delete static_cast<vector<T>*>(v);});
    return py::array_t<T>(count,data,holder);
}

struct Reader {
    std::string products,campaigns;
    Symbols users{'u'},event_ids{'e'};
    vector<Text> category_names;
    vector<std::pair<Text,Text>> campaign_names;
    Input links,events,clicks;
    vector<Buckets<Record>> raw;
    vector<Buckets<Click>> ads;
    std::unique_ptr<Links> identity;
    // This layer decodes strict CSV/timestamps and builds the transitive identity
    // dimension. Polars deduplicates and performs sessionization and attribution.
    Reader(const std::string& folder,I64 start,I64 end,int workers)
        :products(read(fs::path(folder)/"products.csv")),campaigns(read(fs::path(folder)/"campaigns.csv")),
         links(fs::path(folder)/"links"),events(fs::path(folder)/"events"),clicks(fs::path(folder)/"clicks"),
         raw(events.chunks.size()),ads(clicks.chunks.size()) {
        threads=std::clamp(workers,1,64);
        Dict<int> campaigns_by_id,products_by_id;
        category_names={"unknown"};
        rows<2>(products,[&](Text,auto f){category_names.push_back(f[1]);});
        std::sort(category_names.begin(),category_names.end());
        category_names.erase(std::unique(category_names.begin(),category_names.end()),category_names.end());
        Dict<int> categories;
        for (size_t i=0;i<category_names.size();++i) categories[category_names[i]]=int(i);
        rows<2>(products,[&](Text,auto f){products_by_id[f[0]]=categories.at(f[1]);});
        int unknown=categories.at("unknown");
        campaign_names={{"organic","organic"}};
        rows<2>(campaigns,[&](Text,auto f){
            campaigns_by_id[f[0]]=int(campaign_names.size());
            campaign_names.emplace_back(f[0],f[1]);
        });
        size_t bytes=0;
        for (auto& f:links.files) bytes+=f.view().size();
        identity=std::make_unique<Links>(bytes/20,users);
        parallel(events.chunks.size()+1,[&](size_t task){
            if (task==0) {
                for (auto chunk:links.chunks) records<2>(chunk.rows(),[&](Text,auto f){
                    if (!f[0].empty() && !f[1].empty()) identity->join(f[0],f[1]);
                });
                identity->finish();
                return;
            }
            size_t i=task-1;
            auto chunk=events.chunks[i].rows();
            reserve_chunks(raw[i],chunk);
            records<7>(chunk,[&](Text,auto f){
                if (f[0].empty()) return;
                Id id=event_ids.encode(f[0]);
                auto& row=raw[i][bucket(id)].emplace_back(Record{id,ingest_time(f[3]),{}});
                row.event.price=-1;
                if (f[1].empty() || f[5].empty()) return;
                I64 ts=timestamp(f[2]);
                if (ts<start || ts>=end || row.ingest>ts+day_ms) return;
                int price=0,category=unknown;
                if (f[4]=="purchase") {
                    auto value=number(f[6]);
                    if (value<1 || value>10000000) return;
                    price=int(value);
                    auto it=products_by_id.find(f[5]);
                    if (it!=products_by_id.end()) category=it->second;
                } else if (f[4]!="view" && f[4]!="cart") return;
                row.event={users.encode(f[1]),ts,price,category};
            });
        });
        parallel(clicks.chunks.size(),[&](size_t i){
            auto chunk=clicks.chunks[i].rows();
            reserve_chunks(ads[i],chunk);
            records<4>(chunk,[&](Text,auto f){
                if (f[0].empty() || f[1].empty()) return;
                auto campaign=campaigns_by_id.find(f[2]);
                if (campaign==campaigns_by_id.end()) return;
                auto ts=timestamp(f[3]);
                if (ts<start-week || ts>=end) return;
                Id person=identity->person(users.encode(f[1]));
                ads[i][bucket(person)].push_back({person,ts,f[0],campaign->second});
            });
        });
    }

    size_t event_count() const {
        size_t count=0;
        for (const auto& chunk:raw) for (const auto& partition:chunk) count+=partition.size();
        return count;
    }
    py::dict event_part(size_t b) {
        size_t n=0;
        for (auto& chunk:raw) n+=chunk.at(b).size();
        EventColumns c;
        vector<uint8_t> bins_for_users;
        c.id.reserve(n); c.user.reserve(n); c.ts.reserve(n); c.ingest.reserve(n);
        c.price.reserve(n); c.category.reserve(n); bins_for_users.reserve(n);
        for (auto& chunk:raw) {
            for (auto& r:chunk[b]) {
                auto user=r.event.price<0?missing_id:identity->person(r.event.user);
                c.id.push_back(r.id); c.user.push_back(user); c.ingest.push_back(r.ingest);
                c.ts.push_back(r.event.ts); c.price.push_back(r.event.price); c.category.push_back(r.event.category);
                bins_for_users.push_back(uint8_t(bucket(user)));
            }
            vector<Record>().swap(chunk[b]);
        }
        py::dict result;
        result["id"]=array(std::move(c.id)); result["user"]=array(std::move(c.user));
        result["ingest"]=array(std::move(c.ingest)); result["t"]=array(std::move(c.ts));
        result["price"]=array(std::move(c.price)); result["category"]=array(std::move(c.category));
        result["shard"]=array(std::move(bins_for_users));
        return result;
    }

    py::tuple click_part(size_t b) {
        size_t n=0;
        for (auto& chunk:ads) n+=chunk.at(b).size();
        ClickColumns c;
        c.user.reserve(n); c.ts.reserve(n); c.campaign.reserve(n); c.id.reserve(n);
        for (auto& chunk:ads) {
            for (auto& r:chunk[b]) {
                c.user.push_back(r.user);c.ts.push_back(r.ts);c.campaign.push_back(r.campaign);c.id.push_back(r.id);
            }
            vector<Click>().swap(chunk[b]);
        }
        // Rank exact byte strings locally. All clicks of one person are in the
        // same shard, so these ranks preserve every timestamp tie comparison.
        vector<uint64_t> rank(n);
        bool compact=true;
        for (size_t i=0;i<n;++i) {
            auto id=c.id[i];
            if (id.empty() || id[0]!='k' || id.size()>9 || id.back()==0) { compact=false; break; }
            uint64_t word=0;
            std::memcpy(&word,id.data()+1,id.size()-1);
            rank[i]=__builtin_bswap64(word);
        }
        if (!compact) {
            vector<unsigned> order(n);
            std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return c.id[a]<c.id[b];});
            for (size_t i=0;i<n;++i) rank[order[i]]=i;
        }
        return py::make_tuple(array(std::move(c.user)),array(std::move(c.ts)),
                              array(std::move(c.campaign)),array(std::move(rank)));
    }

    vector<std::string> decode(py::array_t<Id,py::array::c_style|py::array::forcecast> ids) {
        vector<std::string> names;
        names.reserve(ids.size());
        std::array<char,9> local;
        for (size_t i=0;i<size_t(ids.size());++i) names.emplace_back(users.decode(ids.data()[i],local));
        return names;
    }
};
PYBIND11_MODULE(shop_reader,m) {
    py::class_<Reader>(m,"Reader")
        .def(py::init<const std::string&,I64,I64,int>())
        .def("event_count",&Reader::event_count)
        .def("event_part",&Reader::event_part)
        .def("click_part",&Reader::click_part)
        .def("decode",&Reader::decode)
        .def_readonly("categories",&Reader::category_names)
        .def_readonly("campaigns",&Reader::campaign_names);
    m.attr("SHARDS")=bins;
}
