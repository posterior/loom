#include <loom/differ.hpp>

namespace loom
{

Differ::Differ (const ValueSchema & schema) :
    schema_(schema),
    unobserved_(get_unobserved(schema)),
    row_count_(0),
    booleans_(schema.booleans_size),
    counts_(schema.counts_size),
    reals_(schema.reals_size),
    tare_()
{
    tare_.mutable_observed()->set_sparsity(ProductValue::Observed::NONE);
    schema_.normalize_dense(* tare_.mutable_observed());
    schema_.validate(tare_);
}

void Differ::set_tare (const ProductValue & tare)
{
    schema_.validate(tare);
    tare_ = tare;
    schema_.normalize_dense(* tare_.mutable_observed());
    schema_.validate(tare_);
}

void Differ::add_rows (const char * rows_in)
{
    protobuf::InFile rows(rows_in);
    protobuf::Row row;
    while (rows.try_read_stream(row)) {
        LOOM_ASSERT(not row.has_diff(), "row is already sparsified");
        const auto & value = row.data();
        LOOM_ASSERT_EQ(
            value.observed().sparsity(),
            ProductValue::Observed::DENSE);

        auto observed = value.observed().dense().begin();
        for (size_t i = 0; i < schema_.booleans_size; ++i) {
            if (*observed++) {
                booleans_[i].add(value.booleans(i));
            }
        }
        for (size_t i = 0; i < schema_.counts_size; ++i) {
            if (*observed++) {
                counts_[i].add(value.counts(i));
            }
        }
        for (size_t i = 0; i < schema_.reals_size; ++i) {
            if (*observed++) {
                reals_[i].add(value.reals(i));
            }
        }
        ++row_count_;
    }

    make_tare();
}

void Differ::make_tare ()
{
    tare_.Clear();
    tare_.mutable_observed()->set_sparsity(ProductValue::Observed::DENSE);

    make_tare_type(booleans_, * tare_.mutable_booleans());
    make_tare_type(counts_, * tare_.mutable_counts());
    make_tare_type(reals_, * tare_.mutable_reals());

    schema_.validate(tare_);
}

void Differ::sparsify_rows (
        const protobuf::Config::Sparsify & config,
        const char * absolute_rows_in,
        const char * relative_rows_out) const
{
    LOOM_ASSERT(config.run(), "sparsify is not configured to run");
    const float sparse_threshold = config.sparse_threshold();
    LOOM_ASSERT_LE(0.0, sparse_threshold);
    LOOM_ASSERT_LE(sparse_threshold, 1.0);

    protobuf::InFile absolute_rows(absolute_rows_in);
    if (absolute_rows.is_file()) {
        LOOM_ASSERT(
            std::string(absolute_rows_in) != std::string(relative_rows_out),
            "in-place sparsify is not supported");
    }
    protobuf::OutFile relative_rows(relative_rows_out);
    protobuf::Row abs;
    protobuf::Row rel;
    while (absolute_rows.try_read_stream(abs)) {
        rel.set_id(abs.id());
        auto & pos = * rel.mutable_data();
        auto & neg = * rel.mutable_diff();
        absolute_to_relative(abs.data(), pos, neg);
        schema_.normalize_small(* pos.mutable_observed(), sparse_threshold);
        schema_.normalize_small(* neg.mutable_observed(), sparse_threshold);
        relative_rows.write_stream(rel);
    }
}

protobuf::ProductValue::Observed Differ::get_unobserved (
        const ValueSchema & schema)
{
    protobuf::ProductValue::Observed unobserved;
    unobserved.set_sparsity(ProductValue::Observed::DENSE);
    for (size_t i = 0; i < schema.total_size(); ++i) {
        unobserved.add_dense(false);
    }
    return unobserved;
}

template<class Summaries, class Values>
inline void Differ::make_tare_type (
        const Summaries & summaries,
        Values & values)
{
    const float count_threshold = 0.5 * row_count_;
    for (const auto & summary : summaries) {
        const auto mode = summary.get_mode();
        bool is_dense = (summary.get_count(mode) > count_threshold);
        tare_.mutable_observed()->add_dense(is_dense);
        if (is_dense) {
            values.Add(mode);
        }
    }
}

template<class T>
inline void Differ::_abs_to_rel (
        const ProductValue & abs,
        ProductValue & pos,
        ProductValue & neg,
        const BlockIterator & block) const
{
    const auto & tare_dense = tare_.observed().dense();
    const auto & abs_dense = abs.observed().dense();
    auto & pos_dense = * pos.mutable_observed()->mutable_dense();
    auto & neg_dense = * neg.mutable_observed()->mutable_dense();

    const auto & tare_values = protobuf::Fields<T>::get(tare_);
    const auto & abs_values = protobuf::Fields<T>::get(abs);
    auto & pos_values = protobuf::Fields<T>::get(pos);
    auto & neg_values = protobuf::Fields<T>::get(neg);

    size_t tare_pos = 0;
    size_t abs_pos = 0;
    for (size_t i = block.begin(); i < block.end(); ++i) {
        const bool tare_observed = tare_dense.Get(i);
        const bool abs_observed = abs_dense.Get(i);
        if (tare_observed) {
            const auto tare_value = tare_values.Get(tare_pos++);
            if (LOOM_LIKELY(abs_observed)) {
                const auto abs_value = abs_values.Get(abs_pos++);
                if (LOOM_UNLIKELY(abs_value != tare_value)) {
                    pos_dense.Set(i, true);
                    pos_values.Add(abs_value);
                    neg_dense.Set(i, true);
                    neg_values.Add(tare_value);
                }
            } else {
                neg_dense.Set(i, true);
                neg_values.Add(tare_value);
            }
        } else {
            if (LOOM_UNLIKELY(abs_observed)) {
                const auto abs_value = abs_values.Get(abs_pos++);
                pos_dense.Set(i, true);
                pos_values.Add(abs_value);
            }
        }
    }
}

template<class T>
inline void Differ::_rel_to_abs (
        ProductValue & abs,
        const ProductValue & pos,
        const ProductValue & neg,
        const BlockIterator & block) const
{
    const auto & tare_dense = tare_.observed().dense();
    auto & abs_dense = * abs.mutable_observed()->mutable_dense();
    const auto & pos_dense = pos.observed().dense();
    const auto & neg_dense = neg.observed().dense();

    const auto & tare_values = protobuf::Fields<T>::get(tare_);
    auto & abs_values = protobuf::Fields<T>::get(abs);
    const auto & pos_values = protobuf::Fields<T>::get(pos);

    size_t tare_pos = 0;
    size_t pos_pos = 0;
    for (size_t i = block.begin(); i < block.end(); ++i) {
        const bool tare_observed = tare_dense.Get(i);
        const bool pos_observed = pos_dense.Get(i);
        if (pos_observed) {
            const auto pos_value = pos_values.Get(pos_pos++);
            abs_dense.Set(i, true);
            abs_values.Add(pos_value);
        }
        if (tare_observed) {
            const auto tare_value = tare_values.Get(tare_pos++);
            if (not neg_dense.Get(i)) {
                LOOM_ASSERT1(not pos_observed, "tare, pos, and neg disagree");
                abs_dense.Set(i, true);
                abs_values.Add(tare_value);
            }
        }
    }
}

void Differ::absolute_to_relative (
        const ProductValue & abs,
        ProductValue & pos,
        ProductValue & neg) const
{
    LOOM_ASSERT_EQ(abs.observed().sparsity(), ProductValue::Observed::DENSE);

    pos.Clear();
    neg.Clear();

    * pos.mutable_observed() = unobserved_;
    * neg.mutable_observed() = unobserved_;

    BlockIterator block;

    block(schema_.booleans_size);
    _abs_to_rel<bool>(abs, pos, neg, block);

    block(schema_.counts_size);
    _abs_to_rel<uint32_t>(abs, pos, neg, block);

    block(schema_.reals_size);
    _abs_to_rel<float>(abs, pos, neg, block);

    if (LOOM_DEBUG_LEVEL >= 3) {
        ProductValue actual;
        relative_to_absolute(actual, pos, neg);
        LOOM_ASSERT_EQ(actual, abs);
    }
}

void Differ::relative_to_absolute (
        ProductValue & abs,
        const ProductValue & pos,
        const ProductValue & neg) const
{
    LOOM_ASSERT_EQ(pos.observed().sparsity(), ProductValue::Observed::DENSE);
    LOOM_ASSERT_EQ(neg.observed().sparsity(), ProductValue::Observed::DENSE);

    abs.Clear();
    * abs.mutable_observed() = unobserved_;

    BlockIterator block;

    block(schema_.booleans_size);
    _rel_to_abs<bool>(abs, pos, neg, block);

    block(schema_.counts_size);
    _rel_to_abs<uint32_t>(abs, pos, neg, block);

    block(schema_.reals_size);
    _rel_to_abs<float>(abs, pos, neg, block);
}

} // namespace loom