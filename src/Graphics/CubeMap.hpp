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
	void load(const std::array<std::string, 6>& paths);

	void create(const std::array<void*, 6>& data,
				size_t width, 
				size_t height,
				int compCount);
	
	virtual void bind(unsigned int unit = 0) const override;
	
	virtual void unbind(unsigned int unit = 0) const override;
private:
};
