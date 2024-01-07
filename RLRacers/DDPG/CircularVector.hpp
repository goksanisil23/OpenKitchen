#pragma once

#include <vector>

template <typename T>
class CircularVector
{
  private:
    std::vector<T> data_;
    size_t         capacity_;
    size_t         head_;

  public:
    CircularVector() = default;

    CircularVector(const size_t capacity) : capacity_{capacity}, head_{0}
    {
        data_.reserve(capacity_);
    }

    void push(const T &value)
    {
        if (data_.size() == capacity_)
        {
            data_[head_] = value;
            head_        = (head_ + 1) % capacity_;
        }
        else
        {
            data_.push_back(value);
        }
    }

    const T &operator[](const size_t index) const
    {
        return data_[index];
    }

    size_t size() const
    {
        return data_.size();
    }

    T *data()
    {
        return data_.data();
    }

    const T *data() const
    {
        return data_.data();
    }
};