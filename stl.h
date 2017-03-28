//
// Created by Martin Wickham on 3/27/17.
//

#ifndef SUBDIV_STL_H
#define SUBDIV_STL_H

#include <vector>
#include <glm/glm.hpp>

struct STLTriangle {
    glm::vec3 normal;
    glm::vec3 verts[3];
    uint16_t flags;
};

/**
 * Loads a binary STL file. All triangles are assumed to have clockwise winding.
 * @param filename
 * @param triangles
 * @param calculateNormals
 * @return
 */
bool loadBinarySTL(const char *filename, std::vector<STLTriangle> &triangles, bool calculateNormals);


#endif //SUBDIV_STL_H
