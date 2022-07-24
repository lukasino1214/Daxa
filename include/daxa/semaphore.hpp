#pragma once

#include <daxa/core.hpp>

namespace daxa
{
    struct BinarySemaphoreInfo
    {
        std::string debug_name = {};
    };

    struct BinarySemaphore : Handle
    {
        ~BinarySemaphore();

        auto info() const -> BinarySemaphoreInfo const &;

      private:
        friend struct Device;
        BinarySemaphore(std::shared_ptr<void> impl);
    };

    struct TimelineSemaphoreInfo
    {
        u64 initial_value = {};
        std::string debug_name = {};
    };

    struct TimelineSemaphore : Handle
    {
        ~TimelineSemaphore();

        auto info() const -> TimelineSemaphoreInfo const &;

        auto value() const -> u64;
        void set_value(u64 value);
        auto wait_for_value(u64 value, u64 timeout_nanos = ~0ull) -> bool;

      private:
        friend struct Device;
        TimelineSemaphore(std::shared_ptr<void> impl);
    };
} // namespace daxa