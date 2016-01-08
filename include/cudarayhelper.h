#ifndef CUDARAYHELPER_H
#define CUDARAYHELPER_H
#include "cudahelper.h"

//#ifndef __CUDA_ARCH__
#include <vector>
//#endif

namespace NPCudaRayHelper
{

	struct Ray
	{
		float3 orig;
		float3 dir;
		__hd__ Ray(float3 o, float3 d) : orig(o), dir(d) {}
	};

	struct Sphere
	{
		float3 center;
		float radius;
		__hd__ Sphere(const float3 c, const float r) : center(c), radius(r) {}
		__device__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal)
		{
			float3 co = r.orig - center;
			float a = r.dir*r.dir;
			if (fabs(a) < M_EPSILON)
				return false;
			float b = 2.f * r.dir*co;
			float c = co*co - radius * radius;
			float root = b * b - 4.f * a * c;
			if (root < 0)
				return false;
			root = sqrtf(root);
			float t0 = fmax((-b + root) / 2.f * a, 0.f);
			float t1 = fmax((-b - root) / 2.f * a, 0.f);
			float t = fmin(t0, t1);
			if (t <= 0)
				return false;
			hitPoint = r.orig + r.dir * t;
			hitNormal = normalize(hitPoint - center);
			return true;
		}
	};


	struct AABBBox
	{
		float3 minPoint;
		float3 maxPoint;
		__hd__ AABBBox(float3 min, float3 max)
			:minPoint(min), maxPoint(max)
		{}

		__device__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal)
		{
			float3 modDir = r.dir;
			modDir.x = escapeZero(modDir.x, M_EPSILON);
			modDir.y = escapeZero(modDir.y, M_EPSILON);
			modDir.z = escapeZero(modDir.z, M_EPSILON);
			float3 tmin = (minPoint - r.orig) / modDir;
			float3 tmax = (maxPoint - r.orig) / modDir;
			float3 real_min = vecmin(tmin, tmax);
			float3 real_max = vecmax(tmin, tmax);
			float minmax = min(min(real_max.x, real_max.y), real_max.z);
			float maxmin = max(max(real_min.x, real_min.y), real_min.z);
			if (minmax >= maxmin && maxmin > M_EPSILON)
			{
				hitPoint = r.orig + r.dir * maxmin;
				hitNormal = (maxmin == real_min.x) ? make_float3(1.f, 0.f, 0.f) :
					(maxmin == real_min.y) ? make_float3(0.f, 1.f, 0.f) : make_float3(0.f, 0.f, 1.f);
				if (hitNormal*r.dir > 0.f)
					hitNormal = -1 * hitNormal;
				return true;
			}
			return false;
		}

	protected:
		__hd__ float escapeZero(const float value, const float epsilon)
		{
			float result = value;
			if (fabs(result) < epsilon)
				result = (result > 0) ? result + epsilon : result - epsilon;
			return result;
		}
		__hd__ float3 vecmin(const float3& lhs, const float3& rhs)
		{
			return make_float3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z));
		}
		__hd__ float3 vecmax(const float3& lhs, const float3& rhs)
		{
			return make_float3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z));
		}
	};

	struct Tri
	{
		// CCW
		float3 p0;
		float3 p1;
		float3 p2;

		float3 n0;
		float3 n1;
		float3 n2;

		float2 uv0;
		float2 uv1;
		float2 uv2;

		__hd__ Tri(float3 a, float3 b, float3 c)
			: p0(a)
			, p1(b)
			, p2(c)
		{

		}

		__device__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal, float &w, float &u, float &v)
		{
			float3 e1 = p1 - p0;
			float3 e2 = p2 - p0;
			if ((e1%e2)*r.dir > 0.f) return false;
			float3 de2 = r.dir%e2;
			float divisor = de2*e1;
			if (fabs(divisor) < M_EPSILON)
				return false;
			float3 t = r.orig - p0;
			float3 te1 = t%e1;
			float rT = (te1*e2) / divisor;
			if (rT < 0.f)
				return false;
			u = de2*t;
			v = te1*r.dir;
			w = 1 - u - v;
			if (u < 0.f || u > 1.f || v > 1.f || v < 0.f || w > 1.f || w < 0.f)
				return false;
			hitPoint = r.orig + rT * r.dir;
			hitNormal = normalize(e1%e2);
			return true;
		}

		__device__ float3 getInterpolatedNormal(float w, float u, float v)
		{
			return normalize(n0 * w + n1 * u + n2 * v);
		}

		__device__ float2 getInterpolatedUV(float w, float u, float v)
		{
			return normalize(uv0 * w + uv1 * u + uv2 * v);
		}
	};

	// ----------------------------------------------------------------------------------------------
	struct Vertex
	{
		float3 pos;
		float3 normal;
		float3 tangent;
		float2 texCoord;
		__hd__ Vertex() : pos(), normal(), tangent(), texCoord() {}
		__host__ Vertex(float3 _pos, float3 _norm, float3 _tan, float2 _tex)
			: pos(_pos), normal(_norm), tangent(_tan), texCoord(_tex) {}
	};

	struct Mesh
	{
		Vertex* vertices;
		uint* indices;
		uint* vertN;
		uint* indN;
		bool isGPUMem;
		__hd__ Mesh() : vertices(0), indices(0), vertN(0), indN(0), isGPUMem(false) {}
		__host__ Mesh(std::vector<Vertex> _verts, std::vector<uint> _indices) : vertices(0), indices(0), vertN(0), indN(0), isGPUMem(false)
		{
			vertN = new uint();
			indN = new uint();
			*vertN = _verts.size();
			*indN = _indices.size();
			vertices = new Vertex[*vertN];
			indices = new uint[*indN];
			for (uint i = 0; i < _verts.size(); i++)
				vertices[i] = _verts[i];
			for (uint i = 0; i < _indices.size(); i++)
				indices[i] = _indices[i];
		}

		__hd__ ~Mesh()
		{
		}

		__host__ bool GenerateDeviceData();
		__host__ bool GetIsGPUMem() const { return isGPUMem; }
		__host__ void Clear() 
		{
			if (isGPUMem)
			{
				if (vertices) cudaFree(vertices);
				if (indices) cudaFree(indices);
				if (vertN) cudaFree(vertN);
				if (indN) cudaFree(indN);
			}
			else
			{
				if (vertices) delete[] vertices;
				if (indices) delete[] indices;
				if (vertN) delete vertN;
				if (indN) delete indN;
			}
		}

		__device__ bool RayIntersectVerts(const float3 p0, const float3 p1, const float3 p2
			, const Ray &r, float &dist, float &_w, float &_u, float &_v)
		{
			float3 e1 = p1 - p0;
			float3 e2 = p2 - p0;
			float3 tvec = r.orig - p0;
			float3 pvec = vecCross(r.dir, e2);
			float  det = vecDot(e1, pvec);
			
			det = 1.0f/ det;

			float u = vecDot(tvec, pvec) * det;

			if (u < 0.0f || u > 1.0f)
				return false;

			float3 qvec = vecCross(tvec, e1);

			float v = vecDot(r.dir, qvec) * det;

			if (v < 0.0f || (u + v) > 1.0f)
				return false;

			_w = 1.f - u - v;
			_u = u;
			_v = v;
			dist = vecDot(e2, qvec) * det;
			if (dist < 0.f)
				return false;
			return true;

			//if ((e1%e2)*r.dir > 0.f) return false;
			//float3 de2 = r.dir%e2; //pvec
			//float divisor = de2*e1; // det
			//if (fabs(divisor) < M_EPSILON)
			//	return false;
			//float3 t = r.orig - p0;
			//float3 te1 = t%e1; // qvec
			//float rT = (te1*e2) / divisor;
			//if (rT < 0.f)
			//	return false;
			//_u = de2*t / divisor;
			//if (_u < 0.0f || _u > 1.0f)
			//		return false;
			//_v = te1*r.dir / divisor;
			//if (_v < 0.0f || (_u + _v) > 1.0f)
			//	return false;
			//_w = 1 - _u - _v;
			//dist = rT;
			//return true;
		}

		__device__ bool Intersect(const Ray &r, uint &triId, float &hitDist, float &w, float &u, float &v)
		{
			float minHit = M_INF;
			for (uint i = 0; i < *indN/3; i++)
			{
				float3 p0 = vertices[indices[i * 3]].pos;
				float3 p1 = vertices[indices[i * 3 + 1]].pos;
				float3 p2 = vertices[indices[i * 3 + 2]].pos;
				float _w, _u, _v;
				if (RayIntersectVerts(p0, p1, p2, r, hitDist, _w, _u, _v))
				{
					if (hitDist < minHit)
					{
						minHit = hitDist;
						w = _w;
						u = _u;
						v = _v;
						triId = i;
					}
				}
			}
			hitDist = minHit;

			return minHit < M_INF;
		}
	};

	struct Model
	{
		__hd__ Model() : devModel(0){}

		std::vector<Mesh*> meshes;
		__host__ bool GenerateDeviceData();
		__host__ bool LoadOBJ(const char* path);
		__host__ void Clear()
		{
			for (uint i = 0; i < meshes.size(); i++)
			{
				meshes[i]->Clear();
				delete meshes[i];
			}
			meshes.clear();
			if (devModel)
			{
				if (devModelData.meshes) cudaFree(devModelData.meshes);
				cudaFree(devModel);
			}
		}

		struct DeviceModel
		{
			Mesh* meshes;
			uint length;
			__hd__ DeviceModel() : meshes(0), length(0) {}

			__device__ bool Intersect(const Ray &r, uint &meshId, uint &triId, float &hitDist, float &w, float &u, float &v)
			{
				float minHit = M_INF;
				float _w, _u, _v;
				uint _triId;
				for (uint i = 0; i < length; i++)
				{
					if (meshes[i].Intersect(r, _triId, hitDist, _w, _u, _v))
					{
						if (hitDist < minHit)
						{
							minHit = hitDist;
							w = _w;
							u = _u;
							v = _v;
							triId = _triId;
							meshId = i;
						}
					}
				}
				hitDist = minHit;

				return minHit < M_INF;
			}
		};
		DeviceModel devModelData;
		DeviceModel* devModel;
	};

	struct Scene
	{
		std::vector<Model*> models;
		__host__ bool GenerateDeviceData();
		__host__ bool AddObjModel(const char* path)
		{
			Model* objModel = new Model();
			if (objModel->LoadOBJ(path))
			{
				models.push_back(objModel);
				return true;
			}
			else
			{
				delete objModel;
			}
			return false;
		}

		struct DeviceScene
		{
			Model::DeviceModel* models;
			uint length;
			__hd__ DeviceScene() : models(0), length(0) {}

			__device__ bool Intersect(const Ray &r,uint &modelId, uint &meshId, uint &triId, float &hitDist, float &w, float &u, float &v)
			{
				float minHit = M_INF;
				for (uint i = 0; i < length; i++)
				{
					if (models[i].Intersect(r, meshId, triId, hitDist, w, u, v))
					{
						if (hitDist < minHit)
						{
							minHit = hitDist;
							modelId = i;
						}
					}
				}
				hitDist = minHit;

				return minHit < M_INF;
			}
		};
		DeviceScene devSceneData;
		DeviceScene* devScene;
	};

}
#endif