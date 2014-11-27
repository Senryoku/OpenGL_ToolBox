#pragma once

#include <glm/glm.hpp>
#define GLM_FORCE_RADIANS
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

class BoundingShape
{
public:
	BoundingShape() =default;
	virtual ~BoundingShape() =default;
};

class BoundingSphere : public BoundingShape
{
public:
	BoundingSphere() =default;
	BoundingSphere(const glm::vec3& _center, float _radius) :
		center(_center),
		radius(_radius)
	{
	}
	
	glm::vec3	center;
	float			radius;
};

class BoundingBox : public BoundingShape
{
public:
	BoundingBox() =default;
	BoundingBox(const glm::vec3& _min, const glm::vec3& _max) :
		min(_min),
		max(_max)
	{
	}
	
	glm::vec3	min;
	glm::vec3	max;
};
