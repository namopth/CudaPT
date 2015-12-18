#ifndef CAMHELPER_H
#define CAMHELPER_H

// Have no time for building my own math lib.
#include <glm/glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "mathhelper.h"

namespace NPCamHelper
{
	class ICamera
	{
	public:
		virtual void UpdateViewMatrix() = 0;
		virtual const float* GetViewMatrix() = 0;
	};

	class RotateCamera : public ICamera
	{
	public:
		RotateCamera(const float radius, const float pitch = 0.f, const float yaw = 0.f)
			: m_bIsViewMatDirty(true)
			, m_fRadius(radius)
			, m_fPitch(pitch)
			, m_fYaw(yaw)
			, m_fYawMin(0.f)
			, m_fYawMax(M_PI*0.49f)
			, m_v3CamPos(0.f,0.f,0.f)
		{
			m_v3CamTarget.x = m_v3CamTarget.y = m_v3CamTarget.z = 0.f;
			m_v3CamUp.x = m_v3CamUp.z = 0.f;
			m_v3CamUp.y = 1.f;
		}

		inline void SetTargetPos(const float x, const float y, const float z)
		{
			m_v3CamTarget.x = x;
			m_v3CamTarget.y = y;
			m_v3CamTarget.z = z;
			m_bIsViewMatDirty = true;
		}

		inline void SetYaw(const float yaw)
		{
			m_fYaw = yaw;
			m_bIsViewMatDirty = true;
		}

		inline void SetPitch(const float pitch)
		{
			m_fPitch = pitch;
			m_bIsViewMatDirty = true;
		}

		inline void SetRadius(const float radius)
		{
			m_fRadius = radius;
			m_bIsViewMatDirty = true;
		}

		inline void AddYaw(const float addYaw)
		{
			m_fYaw += addYaw;
			m_bIsViewMatDirty = true;
		}

		inline void AddPitch(const float addPitch)
		{
			m_fPitch += addPitch;
			m_bIsViewMatDirty = true;
		}

		inline void AddRadius(const float addRadius)
		{
			m_fRadius += addRadius;
			m_bIsViewMatDirty = true;
		}

		inline const float GetYaw() { return m_fYaw; }
		inline const float GetPitch() { return m_fPitch; }
		inline const float GetRadius() { return m_fRadius; }
		inline const glm::vec3 GetDir() 
		{
			glm::vec3 result;
			m_fYaw = glm::clamp(m_fYaw, m_fYawMin, m_fYawMax);
			result.y = -sin(m_fYaw);
			float temp = cos(m_fYaw);
			result.z = -temp * cos(m_fPitch);
			result.x = -temp * sin(m_fPitch);
			return glm::normalize(result);
		}
		inline const NPMathHelper::Vec3 GetPos()
		{
			if (m_bIsViewMatDirty)
			{
				UpdateViewMatrix();
				m_bIsViewMatDirty = false;
			}
			return m_v3CamPos;
		}

		virtual void UpdateViewMatrix()
		{
			m_fYaw = glm::clamp(m_fYaw, m_fYawMin, m_fYawMax);
			float camY = sin(m_fYaw) * m_fRadius;
			float temp = cos(m_fYaw) * m_fRadius;
			float camZ = temp * cos(m_fPitch);
			float camX = temp * sin(m_fPitch);
			m_v3CamPos = NPMathHelper::Vec3(camX, camY, camZ);
			//m_m4ViewMat = glm::lookAt(m_v3CamTarget + glm::vec3(camX, camY, camZ), m_v3CamTarget, m_v3CamUp);
			m_m4ViewMat2 = NPMathHelper::Mat4x4::lookAt(
				NPMathHelper::Vec3(m_v3CamTarget.x + camX, m_v3CamTarget.y + camY, m_v3CamTarget.z + camZ), NPMathHelper::Vec3(m_v3CamTarget.x, m_v3CamTarget.y, m_v3CamTarget.z), NPMathHelper::Vec3(m_v3CamUp.x, m_v3CamUp.y, m_v3CamUp.z));
		}

		virtual const float* GetViewMatrix() 
		{ 
			if (m_bIsViewMatDirty)
			{
				UpdateViewMatrix();
				m_bIsViewMatDirty = false;
			}
			return m_m4ViewMat2.GetDataColumnMajor();
			//return glm::value_ptr(m_m4ViewMat); 
		}

	protected:
		bool m_bIsViewMatDirty;
		float m_fPitch;
		float m_fYaw;
		float m_fYawMin;
		float m_fYawMax;
		float m_fRadius;
		NPMathHelper::Vec3 m_v3CamPos;
		glm::vec3 m_v3CamTarget;
		glm::vec3 m_v3CamUp;
		glm::mat4x4 m_m4ViewMat;
		NPMathHelper::Mat4x4 m_m4ViewMat2;
	};
}
#endif