#include "cudarayhelper.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace NPCudaRayHelper
{
	__host__ bool Mesh::GenerateDeviceData()
	{
		if (isGPUMem || !vertices || !indices || !vertN || !indN)
			return false;

		Vertex* devVerts;
		uint* devIndices;
		uint* devVertN;
		uint* devIndN;

		cudaMalloc((void**)&devVerts, sizeof(Vertex) * *vertN);
		cudaMalloc((void**)&devIndices, sizeof(uint) * *indN);
		cudaMalloc((void**)&devVertN, sizeof(uint));
		cudaMalloc((void**)&devIndN, sizeof(uint));

		cudaMemcpy(devVerts, vertices, sizeof(Vertex) * *vertN, cudaMemcpyHostToDevice);
		cudaMemcpy(devIndices, indices, sizeof(uint) * *indN, cudaMemcpyHostToDevice);
		cudaMemcpy(devVertN, vertN, sizeof(uint), cudaMemcpyHostToDevice);
		cudaMemcpy(devIndN, indN, sizeof(uint), cudaMemcpyHostToDevice);

		delete[] vertices;
		delete[] indices;
		delete vertN;
		delete indN;

		vertices = devVerts;
		indices = devIndices;
		vertN = devVertN;
		indN = devIndN;

		isGPUMem = true;

		return true;
	}

	__host__ bool Model::GenerateDeviceData()
	{
		for (uint i = 0; i < meshes.size(); i++)
			if (!meshes[i]->GetIsGPUMem())
				meshes[i]->GenerateDeviceData();

		if (!devModel | devModelData.length != meshes.size())
		{
			Mesh* devMeshesData = new Mesh[meshes.size()];
			for (uint i = 0; i < meshes.size(); i++)
				devMeshesData[i] = *meshes[i];
			if (devModelData.meshes) cudaFree(devModelData.meshes);
			cudaMalloc((void**)&devModelData.meshes, meshes.size() * sizeof(Mesh));
			cudaMemcpy(devModelData.meshes, devMeshesData, meshes.size() * sizeof(Mesh)
				, cudaMemcpyHostToDevice);
			delete[] devMeshesData;

			devModelData.length = meshes.size();

			if (!devModel) cudaMalloc((void**)&devModel, sizeof(DeviceModel));
			cudaMemcpy(devModel, &devModelData, sizeof(DeviceModel), cudaMemcpyHostToDevice);
			return true;
		}

		return false;
	}

	__host__ void AssimpProcessNode(Model* model, aiNode* node, const aiScene* scene)
	{
		for (uint i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			std::vector<Vertex> verts;
			std::vector<uint> indices;
			for (uint j = 0; j < mesh->mNumVertices; j++)
			{
				Vertex vertex;
				vertex.pos.x = mesh->mVertices[j].x;
				vertex.pos.y = mesh->mVertices[j].y;
				vertex.pos.z = mesh->mVertices[j].z;
				if (mesh->mNormals)
				{
					vertex.normal.x = mesh->mNormals[j].x;
					vertex.normal.y = mesh->mNormals[j].y;
					vertex.normal.z = mesh->mNormals[j].z;
				}
				if (mesh->mTangents)
				{
					vertex.tangent.x = mesh->mTangents[j].x;
					vertex.tangent.y = mesh->mTangents[j].y;
					vertex.tangent.z = mesh->mTangents[j].z;
				}
				if (mesh->mTextureCoords && mesh->mTextureCoords[0])
				{
					vertex.texCoord.x = mesh->mTextureCoords[0][j].x;
					vertex.texCoord.y = mesh->mTextureCoords[0][j].y;
				}
				verts.push_back(vertex);
			}

			for (uint j = 0; j < mesh->mNumFaces; j++)
			{
				aiFace face = mesh->mFaces[j];
				for (uint k = 0; k < face.mNumIndices; k++)
				{
					indices.push_back(face.mIndices[k]);
				}
			}

			Mesh* modelMesh = new Mesh(verts, indices);
			model->meshes.push_back(modelMesh);
		}

		for (uint i = 0; i < node->mNumChildren; i++)
		{
			AssimpProcessNode(model, node->mChildren[i], scene);
		}
	}

	__host__ bool Model::LoadOBJ(const char* path)
	{

		Assimp::Importer importer;
		std::string sPath = path;
		const aiScene* scene = importer.ReadFile(sPath.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs
			| aiProcess_CalcTangentSpace | aiProcess_GenNormals);
		if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			return false;
		}
		std::string dir = sPath.substr(0, sPath.find_last_of('\\'));
		AssimpProcessNode(this, scene->mRootNode, scene);

		return true;
	}

	__host__ bool Scene::GenerateDeviceData()
	{
		for (uint i = 0; i < models.size(); i++)
		{
			models[i]->GenerateDeviceData();
		}

		if (!devScene || devSceneData.length != models.size())
		{
			Model::DeviceModel* tempDevModelsData = new Model::DeviceModel[models.size()];
			for (uint i = 0; i < models.size(); i++)
			{
				tempDevModelsData[i] = models[i]->devModelData;
			}
			if (devSceneData.models) cudaFree(devSceneData.models);
			cudaMalloc((void**)&devSceneData.models, models.size() * sizeof(Model::DeviceModel));
			cudaMemcpy(devSceneData.models, tempDevModelsData, models.size() * sizeof(Model::DeviceModel)
				, cudaMemcpyHostToDevice);
			delete[] tempDevModelsData;
			devSceneData.length = models.size();
			if (!devScene) cudaMalloc((void**)&devScene, sizeof(DeviceScene));
			cudaMemcpy(devScene, &devSceneData, sizeof(DeviceScene), cudaMemcpyHostToDevice);
		}

		return true;
	}
}