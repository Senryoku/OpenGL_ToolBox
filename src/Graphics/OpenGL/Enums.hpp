#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

/** @file
 * Set of more strongly typed (than GLEnum) enums.
**/

/**
 * OpenGL avalaible Primitives
**/
enum Primitive
{
	Points = GL_POINTS, 									///< Points
	LineStrip = GL_LINE_STRIP, 								///< LineStrip
	LineLoop = GL_LINE_LOOP, 								///< LineLoop
	Lines = GL_LINES, 										///< Lines
	LineStripAdjacency = GL_LINE_STRIP_ADJACENCY, 			///< LineStripAdjacency
	LinesAdjacency = GL_LINES_ADJACENCY, 					///< LinesAdjacency
	TriangleStrip = GL_TRIANGLE_STRIP, 						///< TriangleStrip
	TriangleFan = GL_TRIANGLE_FAN, 							///< TriangleFan
	Triangles = GL_TRIANGLES, 								///< Triangles
	TriangleStripAdjacency = GL_TRIANGLE_STRIP_ADJACENCY,	///< TriangleStripAdjacency
	TrianglesAdjacency = GL_TRIANGLES_ADJACENCY,			///< TrianglesAdjacency
	Patches = GL_PATCHES									///< Patches (Tesselation)
};

/**
 * Convenient cast to GLenum
 * @param e Value to cast to GLenum
**/
template<typename Enum>
inline GLenum glenum(Enum e)
{
	return static_cast<GLenum>(e);
}
