#pragma once

#include <iostream>

#include "../DaxaCore.hpp"

#include "math.hpp"

namespace daxa {

	template<std::floating_point T>
	struct TVec2 {

		constexpr T const* data() const { return &x; }

		constexpr T& operator[](u32 index)
		{
			DAXA_ASSERT(index < 2);
			return (&x)[index];
		}

		constexpr const T& operator[](u32 index) const
		{
			DAXA_ASSERT(index < 2);
			return (&x)[index];
		}

		T x{ 0.0f };
		T y{ 0.0f };
	};

	template<std::floating_point T>
	inline constexpr bool operator==(TVec2<T> vecA, TVec2<T> vecB)
	{
		return vecA.x == vecB.x & vecA.y == vecB.y;
	}

	template<std::floating_point T>
	inline constexpr bool operator!=(TVec2<T> vecA, TVec2<T> vecB)
	{
		return vecA.x != vecB.x | vecA.y != vecB.y;
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> operator-(TVec2<T> vec)
	{
		return { -vec.x, -vec.y };
	}

#define DAXA_TVEC2_OPERATOR_IMPL(op)\
    template<std::floating_point T> \
    inline constexpr TVec2<T> operator##op##(TVec2<T> vec, T scalar) \
    { \
        return { \
            vec.x op scalar, \
            vec.y op scalar, \
        }; \
    } \
    template<std::floating_point T> \
    inline constexpr TVec2<T> operator##op##(TVec2<T> a, TVec2<T> b) \
    { \
        return { \
            a.x op b.x, \
            a.y op b.y, \
        }; \
    } \
    template<std::floating_point T> \
    inline constexpr TVec2<T>& operator##op##=(TVec2<T>& vec, T scalar) \
    { \
        vec.x op##= scalar;\
        vec.y op##= scalar;\
        return vec; \
    } \
    template<std::floating_point T> \
    inline constexpr TVec2<T>& operator##op##=(TVec2<T>& a, TVec2<T> b) \
    { \
        a.x op##= b.x;\
        a.y op##= b.y;\
        return a; \
    }

	DAXA_TVEC2_OPERATOR_IMPL(*)
	DAXA_TVEC2_OPERATOR_IMPL(/)
	DAXA_TVEC2_OPERATOR_IMPL(+)
	DAXA_TVEC2_OPERATOR_IMPL(-)

	template<std::floating_point T>
	inline bool hasNANS(const TVec2<T>& vec)
	{
		return std::isnan(vec.x) || std::isnan(vec.y);
	}

	template<std::floating_point T>
	inline TVec2<T> round(TVec2<T> vec)
	{
		return TVec2{ std::round(vec.x), std::round(vec.y) };
	}

	template<std::floating_point T>
	inline TVec2<T> floor(TVec2<T> vec)
	{
		return TVec2{ std::floor(vec.x), std::floor(vec.y) };
	}

	template<std::floating_point T>
	inline TVec2<T> ceil(TVec2<T> vec)
	{
		return TVec2{ std::ceil(vec.x), std::ceil(vec.y) };
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> min(TVec2<T> vecA, TVec2<T> vecB)
	{
		return { std::min(vecA.x, vecB.x), std::min(vecA.y, vecB.y) };
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> max(TVec2<T> vecA, TVec2<T> vecB)
	{
		return { std::max(vecA.x, vecB.x), std::max(vecA.y, vecB.y) };
	}

	template<std::floating_point T>
	inline T length(TVec2<T> vec)
	{
		return std::sqrt(vec.x * vec.x + vec.y * vec.y);
	}

	template<std::floating_point T>
	inline TVec2<T> normalize(TVec2<T> vec)
	{
		DAXA_ASSERT(vec.x != 0.0f || vec.y != 0.0f);
		T invLen = 1.0f / length(vec);
		return { vec.x * invLen, vec.y * invLen };
	}

	template<std::floating_point T>
	inline T distance(TVec2<T> vecA, TVec2<T> vecB)
	{
		return length(vecA - vecB);
	}

	template<std::floating_point T>
	inline constexpr T dot(TVec2<T> vecA, TVec2<T> vecB)
	{
		return (vecA.x * vecB.x + vecA.y * vecB.y);
	}

	template<std::floating_point T>
	inline constexpr T cross(TVec2<T> vecA, TVec2<T> vecB)
	{
		return vecA.x * vecB.y - vecA.y * vecB.x;
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> cross(TVec2<T> vec, T scalar)
	{
		return TVec2<T>(scalar * vec.y, -scalar * vec.x);
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> cross(T scalar, TVec2<T> vec)
	{
		return TVec2<T>(-scalar * vec.y, scalar * vec.x);
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> reflect(TVec2<T> vec, TVec2<T> n)
	{
		return vec - (2.0f * dot(n, vec)) * n;
	}

	template<std::floating_point T>
	inline std::istream& operator>>(std::istream& is, TVec2<T>& vec)
	{
		is >> vec.x >> vec.y;
		return is;
	}

	template<std::floating_point T>
	inline std::ostream& operator<<(std::ostream& os, const TVec2<T>& vec)
	{
		os << '(' << vec.x << ", " << vec.y << ')';
		return os;
	}

	template<std::floating_point T>
	inline TVec2<T> rotate(TVec2<T> vec, T angle)
	{
		T ca = std::cos(angle * (PI_F32 / 180.0f));
		T sa = std::sin(angle * (PI_F32 / 180.0f));
		return { ca * vec.x - sa * vec.y , sa * vec.x + ca * vec.y };
	}

	template<std::floating_point T>
	inline TVec2<T> rotate90(TVec2<T> vec)
	{
		return { -vec.y, vec.x };
	}

	template<std::floating_point T>
	inline TVec2<T> rotate180(TVec2<T> vec)
	{
		return -vec;
	}

	template<std::floating_point T>
	inline TVec2<T> rotate270(TVec2<T> vec)
	{
		return { -vec.y, -vec.x };
	}

	template<std::floating_point T>
	inline T angle(TVec2<T> vec)
	{
		if (vec.y == 0)
			return vec.x < 0 ? 180 : 0;
		else if (vec.x == 0)
			return vec.y < 0 ? 270 : 90;

		if (vec.y > 0)
			if (vec.x > 0)
				return std::atan(vec.y / vec.x) * RAD<T>::value;
			else
				return 180 - std::atan(vec.y / -vec.x) * RAD<T>::value;
		else
			if (vec.x > 0)
				return 360 - std::atan(-vec.y / vec.x) * RAD<T>::value;
			else
				return 180 + std::atan(-vec.y / -vec.x) * RAD<T>::value;
	}

	template<std::floating_point T>
	inline TVec2<T> abs(TVec2<T> vec)
	{
		return { std::abs(vec.x), std::abs(vec.y) };
	}

	template<std::floating_point T>
	inline constexpr bool isPointInRange(TVec2<T> p, TVec2<T> min, TVec2<T> max)
	{
		return
			p.x <= max.x &
			p.y <= max.y &
			p.x >= min.x &
			p.y >= min.y;
	}

	template<std::floating_point T>
	inline constexpr bool isPointInAABB(TVec2<T> p, TVec2<T> aabbCenter, TVec2<T> aabbSize)
	{
		const TVec2<T> relativePointPos = p - aabbCenter;
		aabbSize *= 0.5f;
		return
			relativePointPos.x <= aabbSize.x &
			relativePointPos.y <= aabbSize.y &
			relativePointPos.x >= -aabbSize.x &
			relativePointPos.y >= -aabbSize.y;
	}

	template<std::floating_point T>
	inline constexpr TVec2<T> clamp(const TVec2<T> vec, const TVec2<T> minValue, const TVec2<T> maValuex)
	{
		return max(minValue, min(maValuex, vec));
	}

	using Vec2 = TVec2<f32>;
}