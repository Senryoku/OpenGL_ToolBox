#pragma once

#include <array>
#include <string>

#include <Texture.hpp>

class CubeMap : public Texture
{
public:
	/**
	 * Order : XPOS, XNEG, YPOS, YNEG, ZPOS, ZNEG
	**/
	void load(const std::array<std::string, 6>& Paths);

	void create(const std::array<void*, 6>& Data,
				size_t width, 
				size_t height,
				int compCount);
	
	void bind(int UnitTexture = 0) const;
	
	static void unbind();
private:
};
