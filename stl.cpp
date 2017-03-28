//
// Created by Martin Wickham on 3/27/17.
//

#include <ios>
#include <fstream>
#include "stl.h"

using namespace std;
using namespace glm;

// If we have more than this many triangles, subdivision shouldn't be necessary anyway.
#define MAX_TRIANGLES 65535

// This is declared explicitly rather than with sizeof(STLTriangle) because STLTriangle may have alignment padding
#define TRIANGLE_SIZE (4 * 3 * sizeof(float) + sizeof(uint16_t))

bool loadBinarySTL(const char *filename, vector<STLTriangle> &triangles, bool calculateNormals) {
    ifstream input(filename, ios::binary | ios::in);
    if (!input) {
        printf("Failed to open file %s\n", filename);
        return false;
    }

    // ASCII STL files start with "SOLID". Make sure this is not an ascii file.
    char header[6];
    header[5] = '\0';
    input.seekg(0);
    input.read(header, 5);
    if (strcmp(header, "SOLID") == 0) {
        printf("%s cannot be loaded because it is an ASCII STL file.\n", filename);
        return false;
    }

    input.seekg(80); // skip the 80 byte header
    uint32_t numTriangles; // read the number of triangles
    input.read((char *)&numTriangles, 4);

    if (numTriangles <= 0) {
        printf("Can't load %d triangles from %s\n", numTriangles, filename);
        return false;
    }

    if (numTriangles > MAX_TRIANGLES) {
        printf("Too many triangles in %s, found %d but max is %d.\n", filename, numTriangles, MAX_TRIANGLES);
        return false;
    }

    triangles.resize(numTriangles);
    for (int c = 0; c < numTriangles; c++) {
        STLTriangle *triangle = &triangles[c];
        input.read((char *)triangle, TRIANGLE_SIZE);
        if (!input) {
            printf("%s is too short to contain all of its triangles.\n", filename);
            return false;
        }
        if (!calculateNormals) continue;

        vec3 normal = cross(triangle->verts[0] - triangle->verts[2], triangle->verts[1] - triangle->verts[2]);
        if (triangle->normal != vec3(0) && dot(normal, triangle->normal) < 0) {
            printf("Warning: backwards normal on triangle %d.\n", c);
            // normal = -normal;
        }
        triangle->normal = normalize(normal);
    }

    auto pos = input.tellg();
    input.seekg(0, ios::end);
    if (pos != input.tellg()) {
        printf("%s contains more data than was loaded.\n", filename);
        // not a fatal error, we can continue.
    }

    return true;
}
