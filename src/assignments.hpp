#pragma once

#include "common.hpp"
#include <vector>
#include <deque>
#include <utility>

namespace loom
{

class Assignments : noncopyable
{
public:

    template<class T>
    class Queue
    {
    public:

        size_t size () const { return queue_.size(); }

        const T & front () const { return queue_.front(); }
        const T & back () const { return queue_.back(); }
        const T & operator[] (size_t i) const { return queue_[i]; }

        void clear () { queue_.clear(); }

        void push (const T & t) { queue_.push_back(t); }

        bool try_push (const T & t)
        {
            if (LOOM_LIKELY(t != queue_.front())) {
                queue_.push_back(t);
                return true;
            } else {
                return false;
            }
        }

        T pop ()
        {
            const T t = queue_.front();
            queue_.pop_front();
            return t;
        }

    private:

        std::deque<T> queue_;
    };

    typedef uint64_t Key;
    typedef uint32_t Value;

    Assignments (size_t dim) : values_(dim) {}

    size_t dim () const { return values_.size(); }
    size_t size () const { return keys_.size(); }

    void clear ();
    void init (size_t dim);
    void load (const char * filename);
    void dump (const char * filename) const;

    Queue<Key> & rowids () { return keys_; }
    Queue<Value> & groupids (size_t i) { return values_[i]; }

private:

    Queue<Key> keys_;
    std::vector<Queue<Value>> values_;
};

} // namespace loom