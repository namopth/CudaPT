#ifndef MATHHELPER_H
#define MATHHELPER_H

#include <math.h>

#define M_E         2.71828182845904523536028747135266250   /* e */
#define M_LOG2E     1.44269504088896340735992468100189214   /* log 2e */
#define M_LOG10E    0.434294481903251827651128918916605082  /* log 10e */
#define M_LN2       0.693147180559945309417232121458176568  /* log e2 */
#define M_LN10      2.30258509299404568401799145468436421   /* log e10 */
#define M_PI        3.14159265358979323846264338327950288   /* pi */
#define M_PI_2      1.57079632679489661923132169163975144   /* pi/2 */
#define M_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
#define M_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
#define M_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
#define M_2_SQRTPI  1.12837916709551257389615890312154517   /* 2/sqrt(pi) */
#define M_SQRT2     1.41421356237309504880168872420969808   /* sqrt(2) */
#define M_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */
#define M_EPSILON	1E-9

#define MAT_COLUMN_MAJOR

namespace NPMathHelper
{
	class Mat4x4;
	class Vec4;
	class Vec3;

	class Vec2
	{
	public:
#pragma pack(push, 1)
		union
		{
			float _e[2];
			struct{
				float _x, _y;
			};
		};
#pragma pack(pop)

		Vec2(const float x = 0.f, const float y = 0.f)
		{
			_x = x; _y = y;
		}

		inline static float dot(const Vec2& v2Left, const Vec2& v2Right)
		{
			return v2Left._x * v2Right._x + v2Left._y * v2Right._y;
		}

		inline static float length(const Vec2& v2Left)
		{
			return sqrt(v2Left._x * v2Left._x + v2Left._y * v2Left._y);
		}

		inline static Vec2 normalize(const Vec2& v2Left);

		inline float dot(const Vec2& other)
		{
			return dot(*this, other);
		}

		inline float length()
		{
			return length(*this);
		}

		inline Vec2 normalize()
		{
			return normalize(*this);
		}

		inline const bool operator == (const Vec2& v2Other) const
		{
			return (_x == v2Other._x && _y == v2Other._y);
		}

		inline const bool operator != (const Vec2& v2Other) const
		{
			return !(_x == v2Other._x && _y == v2Other._y);
		}
	};

	inline const Vec2 operator + (const Vec2& lhs, const float value)
	{
		return Vec2(lhs._x + value, lhs._y + value);
	}

	inline const Vec2 operator + (const float value, const Vec2& rhs)
	{
		return rhs + value;
	}

	inline const Vec2 operator - (const Vec2& lhs, const float value)
	{
		return Vec2(lhs._x - value, lhs._y - value);
	}

	inline const Vec2 operator - (const float value, const Vec2& rhs)
	{
		return rhs - value;
	}

	inline const Vec2 operator * (const Vec2& lhs, const float value)
	{
		return Vec2(lhs._x * value, lhs._y * value);
	}

	inline const Vec2 operator * (const float value, const Vec2& rhs)
	{
		return rhs * value;
	}

	inline const Vec2 operator / (const Vec2& lhs, const float value)
	{
		return Vec2(lhs._x / value, lhs._y / value);
	}

	inline const Vec2 operator / (const float value, const Vec2& rhs)
	{
		return Vec2(value / rhs._x, value / rhs._y);
	}

	inline const Vec2 operator + (const Vec2& lhs, const Vec2& rhs)
	{
		return Vec2(lhs._x + rhs._x, lhs._y + rhs._y);
	}

	inline const Vec2 operator - (const Vec2& lhs, const Vec2& rhs)
	{
		return Vec2(lhs._x - rhs._x, lhs._y - rhs._y);
	}

	inline Vec2 Vec2::normalize(const Vec2& v2Left)
	{
		float l = length(v2Left);
		return Vec2(v2Left) / l;
	}

	class Vec3
	{
	public:
#pragma pack(push, 1)
		union
		{
			float _e[3];
			struct{
				float _x, _y, _z;
			};
			struct{
				Vec2 _v20;
				float _2;
			};
		};
#pragma pack(pop)

		Vec3(const float x = 0.f, const float y = 0.f, const float z = 0.f)
		{
			_x = x;
			_y = y;
			_z = z;
		}
		Vec3(const Vec3& v3Other) { _x = v3Other._x; _y = v3Other._y; _z = v3Other._z; }

		inline static Vec3 cross(const Vec3& v3Left, const Vec3& v3Right)
		{
			return Vec3(v3Left._y*v3Right._z - v3Left._z*v3Right._y,
				v3Left._z*v3Right._x - v3Left._x*v3Right._z,
				v3Left._x*v3Right._y - v3Left._y*v3Right._x);
		}

		inline static float dot(const Vec3& v3Left, const Vec3& v3Right)
		{
			return v3Left._x * v3Right._x + v3Left._y * v3Right._y + v3Left._z * v3Right._z;
		}

		inline static float length(const Vec3& v3Left)
		{
			return sqrt(v3Left._x * v3Left._x + v3Left._y * v3Left._y + v3Left._z * v3Left._z);
		}

		inline static Vec3 normalize(const Vec3& v3Left);

		static Vec3 transform(const Mat4x4& mat4x4Left, const Vec3& v3Right, bool pos = true);

		inline Vec3 cross(const Vec3& v3Other)
		{
			return cross(*this, v3Other);
		}

		inline float dot(const Vec3& v3Other)
		{
			return dot(*this, v3Other);
		}

		inline float length()
		{
			return length(*this);
		}

		inline Vec3 normalize()
		{
			return normalize(*this);
		}

		inline Vec3 transform(const Mat4x4& mat4Other, bool pos = true)
		{
			return transform(mat4Other, *this, pos);
		}

		inline const bool operator == (const Vec3& v3Other) const
		{
			return (_x == v3Other._x && _y == v3Other._y &&_z == v3Other._z);
		}

		inline const bool operator != (const Vec3& v3Other) const
		{
			return !(_x == v3Other._x && _y == v3Other._y &&_z == v3Other._z);
		}
	};

	inline const Vec3 operator + (const Vec3& lhs, const float value)
	{
		return Vec3(lhs._x + value, lhs._y + value, lhs._z + value);
	}

	inline const Vec3 operator + (const float value, const Vec3& rhs)
	{
		return rhs + value;
	}

	inline const Vec3 operator - (const Vec3& lhs, const float value)
	{
		return Vec3(lhs._x - value, lhs._y - value, lhs._z - value);
	}

	inline const Vec3 operator - (const float value, const Vec3& rhs)
	{
		return rhs - value;
	}

	inline const Vec3 operator * (const Vec3& lhs, const float value)
	{
		return Vec3(lhs._x * value, lhs._y * value, lhs._z * value);
	}

	inline const Vec3 operator * (const float value, const Vec3& rhs)
	{
		return rhs * value;
	}

	inline const Vec3 operator / (const Vec3& lhs, const float value)
	{
		return Vec3(lhs._x / value, lhs._y / value, lhs._z / value);
	}

	inline const Vec3 operator / (const float value, const Vec3& rhs)
	{
		return Vec3(value / rhs._x, value / rhs._y, value / rhs._z);
	}

	inline const Vec3 operator + (const Vec3& lhs, const Vec3& rhs)
	{
		return Vec3(lhs._x + rhs._x, lhs._y + rhs._y, lhs._z + rhs._z);
	}

	inline const Vec3 operator - (const Vec3& lhs, const Vec3& rhs)
	{
		return Vec3(lhs._x - rhs._x, lhs._y - rhs._y, lhs._z - rhs._z);
	}

	inline Vec3 Vec3::normalize(const Vec3& v3Left)
	{
		float l = length(v3Left);
		return Vec3(v3Left) / l;
	}

	class Vec4
	{
	public:
#pragma pack(push, 1)
		union
		{
			float _e[4];
			struct{
				float _x, _y, _z, _w;
			};
			struct{
				Vec2 _v20;
				Vec2 _v21;
			};
			struct{
				Vec3 _v30;
				float _3;
			};
		};
#pragma pack(pop)

		Vec4(const float x = 0.f, const float y = 0.f, const float z = 0.f, const float w = 0.f)
		{
			_x = x;
			_y = y;
			_z = z;
			_w = w;
		}

		Vec4(const Vec3& xyz, float w)
			: _v30(xyz), _3(w) { }

		Vec4(const Vec4& v4Other) { _x = v4Other._x; _y = v4Other._y; _z = v4Other._z; _w = v4Other._w; }

		inline static float dot(const Vec4& v4Left, const Vec4& v4Right)
		{
			return v4Left._x * v4Right._x + v4Left._y * v4Right._y + v4Left._z * v4Right._z + v4Left._w * v4Right._w;
		}

		inline static float length(const Vec4& v4Left)
		{
			return sqrt(v4Left._x * v4Left._x + v4Left._y * v4Left._y + v4Left._z * v4Left._z + v4Left._w * v4Left._w);
		}

		inline static Vec4 normalize(const Vec4& v4Left);

		static Vec4 transform(const Mat4x4& mat4x4Left, const Vec4& v4Right);

		inline float dot(const Vec4& v4Other)
		{
			return dot(*this, v4Other);
		}

		inline float length()
		{
			return length(*this);
		}

		inline Vec4 normalize()
		{
			return normalize(*this);
		}

		inline Vec4 transform(const Mat4x4& mat4Other)
		{
			return transform(mat4Other, *this);
		}

		inline const bool operator == (const Vec4& v4Other) const
		{
			return (_x == v4Other._x && _y == v4Other._y &&_z == v4Other._z &&_w == v4Other._w);
		}

		inline const bool operator != (const Vec4& v4Other) const
		{
			return !(_x == v4Other._x && _y == v4Other._y &&_z == v4Other._z &&_w == v4Other._w);
		}
	};

	inline const Vec4 operator + (const Vec4& lhs, const float value)
	{
		return Vec4(lhs._x + value, lhs._y + value, lhs._z + value, lhs._w + value);
	}

	inline const Vec4 operator + (const float value, const Vec4& rhs)
	{
		return rhs + value;
	}

	inline const Vec4 operator - (const Vec4& lhs, const float value)
	{
		return Vec4(lhs._x - value, lhs._y - value, lhs._z - value, lhs._w - value);
	}

	inline const Vec4 operator - (const float value, const Vec4& rhs)
	{
		return rhs - value;
	}

	inline const Vec4 operator * (const Vec4& lhs, const float value)
	{
		return Vec4(lhs._x * value, lhs._y * value, lhs._z * value, lhs._w * value);
	}

	inline const Vec4 operator * (const float value, const Vec4& rhs)
	{
		return rhs * value;
	}

	inline const Vec4 operator * (const Mat4x4& lhs, const Vec4& rhs)
	{
		return Vec4::transform(lhs, rhs);
	}

	inline const Vec4 operator * (const Vec4& lhs, const Mat4x4& rhs)
	{
		return rhs * lhs;
	}

	inline const Vec4 operator / (const Vec4& lhs, const float value)
	{
		return Vec4(lhs._x / value, lhs._y / value, lhs._z / value, lhs._w / value);
	}

	inline const Vec4 operator + (const Vec4& lhs, const Vec4& rhs)
	{
		return Vec4(lhs._x + rhs._x, lhs._y + rhs._y, lhs._z + rhs._z, lhs._w + rhs._w);
	}

	inline const Vec4 operator - (const Vec4& lhs, const Vec4& rhs)
	{
		return Vec4(lhs._x - rhs._x, lhs._y - rhs._y, lhs._z - rhs._z, lhs._w - rhs._w);
	}

	inline Vec4 Vec4::normalize(const Vec4& v4Left)
	{
		float l = length(v4Left);
		return Vec4(v4Left) / l;
	}

	class Quat
	{
	public:
#pragma pack(push, 1)
		union
		{
			float _e[4];
			struct{
				float _x, _y, _z, _w;
			};
			struct{
				Vec4 _v;
			};
		};
#pragma pack(pop)
		Quat()
		{
			_x = 0.f; _y = 0.f; _z = 0.f; _w = 0.f;
		}

		Quat(const float x, const float y, const float z, const float w)
		{
			_x = x; _y = y; _z = z; _w = w;
		}

		Quat(const Vec4 v)
		{
			_v = v;
		}
	protected:
	};

	class Mat4x4
	{
	public:
#pragma pack(push, 1)
		union
		{
			float _e[16];
#ifdef MAT_COLUMN_MAJOR
			struct{
				float _00, _10, _20, _30;
				float _01, _11, _21, _31;
				float _02, _12, _22, _32;
				float _03, _13, _23, _33;
			};
#else
			struct{
				float _00, _01, _02, _03;
				float _10, _11, _12, _13;
				float _20, _21, _22, _23;
				float _30, _31, _32, _33;
			};
#endif
			struct{
				Vec4 _v40;
				Vec4 _v41;
				Vec4 _v42;
				Vec4 _v43;
			};
		};
#pragma pack(pop)

		Mat4x4(float f00 = 1.f, float f01 = 0.f, float f02 = 0.f, float f03 = 0.f,
			float f10 = 0.f, float f11 = 1.f, float f12 = 0.f, float f13 = 0.f,
			float f20 = 0.f, float f21 = 0.f, float f22 = 1.f, float f23 = 0.f,
			float f30 = 0.f, float f31 = 0.f, float f32 = 0.f, float f33 = 1.f)
		{
			_00 = f00; _01 = f01; _02 = f02, _03 = f03;
			_10 = f10; _11 = f11; _12 = f12, _13 = f13;
			_20 = f20; _21 = f21; _22 = f22, _23 = f23;
			_30 = f30; _31 = f31; _32 = f32, _33 = f33;
		}

		Mat4x4(const float* e)
		{
			for (unsigned int i = 0; i < 16; i++)
				_e[i] = e[i];
		}

#ifdef MAT_COLUMN_MAJOR
		inline float* GetDataColumnMajor() { return _e; }
#else
#endif

		inline static Mat4x4 transpose(const Mat4x4& m4Left)
		{
			Mat4x4 result(m4Left);
			swap(result._01, result._10);
			swap(result._02, result._20);
			swap(result._03, result._30);
			swap(result._12, result._21);
			swap(result._13, result._31);
			swap(result._23, result._32);
			return result;
		}

		static Mat4x4 inverse(const Mat4x4& m4Left);

		inline static Mat4x4 mul(const Mat4x4& m4Left, const Mat4x4& m4Right)
		{
			Mat4x4 tLeft = transpose(m4Left);

			return Mat4x4(Vec4::dot(tLeft._v40, m4Right._v40), Vec4::dot(tLeft._v40, m4Right._v41), 
				Vec4::dot(tLeft._v40, m4Right._v42), Vec4::dot(tLeft._v40, m4Right._v43), 
				Vec4::dot(tLeft._v41, m4Right._v40), Vec4::dot(tLeft._v41, m4Right._v41), 
				Vec4::dot(tLeft._v41, m4Right._v42), Vec4::dot(tLeft._v41, m4Right._v43), 
				Vec4::dot(tLeft._v42, m4Right._v40), Vec4::dot(tLeft._v42, m4Right._v41), 
				Vec4::dot(tLeft._v42, m4Right._v42), Vec4::dot(tLeft._v42, m4Right._v43), 
				Vec4::dot(tLeft._v43, m4Right._v40), Vec4::dot(tLeft._v43, m4Right._v41), 
				Vec4::dot(tLeft._v43, m4Right._v42), Vec4::dot(tLeft._v43, m4Right._v43));
		}

		inline static Mat4x4 Identity()
		{
			return Mat4x4(1.0f, 0.f, 0.f, 0.f,
				0.f, 1.0f, 0.f, 0.f,
				0.f, 0.f, 1.f, 0.f,
				0.f, 0.f, 0.f, 1.f);
		}

		inline static Mat4x4 translation(const Vec3& pos)
		{
			Mat4x4 result = Identity();
			result._v43 = Vec4(pos, 1.0f);
			return result;
		}

		inline static Mat4x4 coordinateTransformation(const Vec3& axis0, const Vec3& axis1, const Vec3& axis2)
		{
			return Mat4x4(axis0._x, axis0._y, axis0._z, 0.f,
				axis1._x, axis1._y, axis1._z, 0.f, 
				axis2._x, axis2._y, axis2._z, 0.f, 
				0.f, 0.f, 0.f, 1.0f);
		}

		inline static Mat4x4 perspectiveProjection(const float fov, const float aspect, const float near, const float far)
		{
			float tangent = tan(fov * 0.5f);
			return Mat4x4(1.0f/(tangent * aspect), 0.f, 0.f, 0.f,
				0.f, 1.0f/tangent, 0.f, 0.f,
				0.f ,0.f , -(near+far)/(far-near), -(2.f*near*far)/(far-near),
				0.f, 0.f, -1.f, 0.f);
		}

		inline static Mat4x4 orthogonalProjection(const float width, const float height, const float near, const float far)
		{
			return Mat4x4(2.f / width, 0.f, 0.f, 0.f,
				0.f, 2.f / height, 0.f, 0.f,
				0.f, 0.f, 2.f / (near - far), (near + far) / (near - far),
				0.f, 0.f, 0.f, 1.f);
		}

		inline static Mat4x4 lookAt(const Vec3& eyePos, const Vec3& eyeTarget, const Vec3& eyeUp)
		{
			Vec3 dir = Vec3::normalize(eyeTarget - eyePos);
			Vec3 right = Vec3::normalize(Vec3::cross(dir, eyeUp));
			Vec3 up = Vec3::cross(right, dir);
			return Mat4x4(right._x, right._y, right._z, -Vec3::dot(right, eyePos),
				up._x, up._y, up._z, -Vec3::dot(up, eyePos),
				-dir._x, -dir._y, -dir._z, Vec3::dot(dir, eyePos),
				0.f, 0.f, 0.f, 1.f);
		}

		inline static Mat4x4 scaleTransform(const float scaleX, const float scaleY, const float scaleZ)
		{
			return Mat4x4(scaleX, 0.f, 0.f, 0.f,
				0.f, scaleY, 0.f, 0.f,
				0.f, 0.f, scaleZ, 0.f,
				0.f, 0.f, 0.f, 1.f);
		}

		inline static Mat4x4 rotationTransform(const Quat& q)
		{
			Mat4x4 result = Identity();

			float x2 = q._x * q._x;
			float y2 = q._y * q._y;
			float z2 = q._z * q._z;
			float xy = q._x * q._y;
			float xz = q._x * q._z;
			float xw = q._x * q._w;
			float yz = q._y * q._z;
			float yw = q._y * q._w;
			float zw = q._z * q._w;

			result._00 = 1.f - 2.f * y2 - 2.f * z2;
			result._01 =  2.f * xy - 2.f * zw;
			result._02 = 2.f * xz + 2.f * yw;

			result._10 = 2.f * xy + 2.f * zw;
			result._11 = 1.f - 2.f * x2 - 2.f * z2;
			result._12 = 2.f * yz - 2.f * xw;

			result._20 = 2.f * xz - 2.f * yw;
			result._21 = 2.f * yz + 2.f * xw;
			result._22 = 1.f - 2.f * x2 - 2.f * y2;

			return result;
		}

		inline const bool operator == (const Mat4x4& matOther) const
		{
			return (_v40 == matOther._v40 && _v41 == matOther._v41 &&_v42 == matOther._v42 &&_v43 == matOther._v43);
		}

		inline const bool operator != (const Mat4x4& matOther) const
		{
			return !(_v40 == matOther._v40 && _v41 == matOther._v41 &&_v42 == matOther._v42 &&_v43 == matOther._v43);
		}

	protected:
		inline static void swap(float& fLeft, float& fRight)
		{
			float temp = fLeft;
			fLeft = fRight;
			fRight = temp;
		}
	};

	inline const Mat4x4 operator + (const Mat4x4& lhs, const float value)
	{
		Mat4x4 result;
		for (unsigned int i = 0; i < 16; i++)
			result._e[i] = lhs._e[i] + value;
		return result;
	}

	inline const Mat4x4 operator + (const float value, const Mat4x4& rhs)
	{
		return rhs + value;
	}

	inline const Mat4x4 operator * (const Mat4x4& lhs, const Mat4x4& rhs)
	{
		return Mat4x4::mul(lhs, rhs);
	}
}

#endif