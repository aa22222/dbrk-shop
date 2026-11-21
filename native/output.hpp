#pragma once
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <filesystem>
#include <string_view>
#include <vector>

using Text = std::string_view;
using I64 = int64_t;
struct UserRow {
    Text user;
    int32_t day;
    I64 sessions{}, events{}, revenue{};
    Text top;
};
struct CampaignRow {
    Text campaign, channel;
    int32_t day;
    I64 sessions{}, purchasing{}, revenue{}, assists{};
};

inline void checked(arrow::Status status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
}
struct Columns {
    arrow::FieldVector fields;
    arrow::ArrayVector arrays;
    template<class Builder, class Row, class Value>
    void add(const char* name, const std::vector<Row>& rows, Value Row::*member, bool nullable=false) {
        Builder builder;
        checked(builder.Reserve(rows.size()));
        if constexpr (std::is_same_v<Value, Text>) {
            int64_t bytes=0;
            for (auto& row:rows) bytes+=(row.*member).size();
            checked(builder.ReserveData(bytes));
        }
        for (auto& row : rows) {
            if constexpr (std::is_same_v<Value, Text>) {
                if (nullable && (row.*member).data() == nullptr) builder.UnsafeAppendNull();
                else builder.UnsafeAppend(row.*member);
            } else builder.UnsafeAppend(row.*member);
        }
        arrays.push_back(builder.Finish().ValueOrDie());
        fields.push_back(arrow::field(name, arrays.back()->type(), nullable));
    }
    void save(const std::filesystem::path& path) {
        auto sink = arrow::io::FileOutputStream::Open(path.string()).ValueOrDie();
        auto table = arrow::Table::Make(arrow::schema(fields), arrays);
        static const auto props=parquet::WriterProperties::Builder().disable_statistics()->build();
        checked(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, 65536, props));
        checked(sink->Close());
    }
};
inline void save(const std::filesystem::path& path, const std::vector<UserRow>& rows) {
    Columns c;
    c.add<arrow::StringBuilder>("user_id", rows, &UserRow::user);
    c.add<arrow::Date32Builder>("day", rows, &UserRow::day);
    c.add<arrow::Int64Builder>("sessions", rows, &UserRow::sessions);
    c.add<arrow::Int64Builder>("events", rows, &UserRow::events);
    c.add<arrow::Int64Builder>("revenue_cents", rows, &UserRow::revenue);
    c.add<arrow::StringBuilder>("top_category", rows, &UserRow::top, true);
    c.save(path);
}
inline void save(const std::filesystem::path& path, const std::vector<CampaignRow>& rows) {
    Columns c;
    c.add<arrow::StringBuilder>("campaign_id", rows, &CampaignRow::campaign);
    c.add<arrow::StringBuilder>("channel", rows, &CampaignRow::channel);
    c.add<arrow::Date32Builder>("day", rows, &CampaignRow::day);
    c.add<arrow::Int64Builder>("sessions", rows, &CampaignRow::sessions);
    c.add<arrow::Int64Builder>("purchasing_sessions", rows, &CampaignRow::purchasing);
    c.add<arrow::Int64Builder>("revenue_cents", rows, &CampaignRow::revenue);
    c.add<arrow::Int64Builder>("assisted_sessions", rows, &CampaignRow::assists);
    c.save(path);
}
