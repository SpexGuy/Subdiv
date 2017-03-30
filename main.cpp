#include <iostream>
#include <cstdlib>

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <stb/stb_image.h>
#include "gl_includes.h"
#include "Perf.h"
#include "stl.h"

using namespace std;
using namespace glm;

GLuint compileShader(const char *vertSrc, const char *fragSrc);
void loadTexture(GLuint texname, const char *filename);

GLFWwindow *window;

mat4 rotation; // identity
mat4 view;
mat4 projection;

struct {
    GLuint normalMat;
    GLuint mvp;
} uniforms;

const char *vert = GLSL(
        const vec3 lightDir = normalize(vec3(1,1,1));
        const float ambient = 0.2;
        const vec3 color = vec3(0.5, 0, 1);

        uniform mat3 normalMat;
        uniform mat4 mvp;

        in vec3 position;
        in vec3 normal;

        flat out vec3 litColor;

        void main() {
            // lighting is not realistic. Meant just to distinguish the faces.
            gl_Position = mvp * vec4(position, 1.0);
            vec3 normal = normalize(normalMat * normal);
            float light = dot(normal, lightDir) / 2 + 0.5;
            light = light + ambient * (1 - light);
            litColor = light * color;
        }
);

const char *frag = GLSL(
        flat in vec3 litColor;
        out vec4 fragColor;

        void main() {
            fragColor = vec4(litColor, 1);
        }
);

struct {
    GLuint mvp;
} lineUniforms;

const char *linevert = GLSL(
        const vec3 color = vec3(0, 1, 0);

        uniform mat4 mvp;

        in vec3 position;

        flat out vec3 litColor;

        void main() {
            // lighting is not realistic. Meant just to distinguish the faces.
            gl_Position = mvp * vec4(position, 1.0);
            litColor = color;
        }
);

GLuint vao, linevao;
GLuint shader, lineShader;
GLuint tribuf, linebuf, lineelm;
GLuint linesSize;

#define BAD_OPP (std::numeric_limits<uint32_t>::max())

struct HalfEdge {
    uint32_t pos;
    uint32_t opp;
};

struct Quad {
    HalfEdge e[4];
};

bool drawAdjacency = false;
bool isTriangles;
vector<Quad> mesh;
vector<vec3> positions;

struct Vertex {
    vec3 pos;
    vec3 nor;

    Vertex() {}
    Vertex(const vec3 &pos, const vec3 &nor) : pos(pos), nor(nor) {}
};

vector<Vertex> verts;


void generateLineBuffer() {
    vector<vec3> pts;
    pts.reserve(mesh.size());
    for (uint32_t c = 0, n = mesh.size(); c < n; c++) {
        vec3 pt = positions[mesh[c].e[0].pos] +
                  positions[mesh[c].e[1].pos] +
                  positions[mesh[c].e[2].pos];
        uint32_t fourth = mesh[c].e[3].pos;
        if (fourth != BAD_OPP) {
            pts.push_back((pt + positions[fourth]) / 4.f);
        } else {
            pts.push_back(pt / 3.f);
        }
    }

    vector<uint32_t> indices;
    indices.reserve(mesh.size() * 4);
    HalfEdge *edge = reinterpret_cast<HalfEdge *>(mesh.data());
    for (uint32_t c = 0, n = mesh.size() * 4; c < n; c++) {
        if (edge[c].opp != BAD_OPP && edge[c].opp > c) {
            indices.push_back(c / 4);
            indices.push_back(edge[c].opp / 4);
        }
    }

    glBindVertexArray(linevao);
    glBindBuffer(GL_ARRAY_BUFFER, linebuf);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(pts[0]), pts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineelm);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_DYNAMIC_DRAW);
    linesSize = indices.size();

    printf("Generated %d indices\n", linesSize);
}

void generateMeshBuffer() {
    verts.clear();
    verts.reserve(mesh.size() * 6);
    for (uint32_t c = 0, n = mesh.size(); c < n; c++) {
        Quad &q = mesh[c];
        vec3 p0 = positions[q.e[0].pos];
        vec3 p1 = positions[q.e[1].pos];
        vec3 p2 = positions[q.e[2].pos];
        if (q.e[3].pos == BAD_OPP) {
            vec3 nor = cross(p0 - p2, p1 - p2);
            verts.emplace_back(p0, nor);
            verts.emplace_back(p1, nor);
            verts.emplace_back(p2, nor);
        } else {
            vec3 p3 = positions[q.e[3].pos];
            vec3 nor = cross(p2 - p0, p3 - p1);
            verts.emplace_back(p1, nor);
            verts.emplace_back(p2, nor);
            verts.emplace_back(p3, nor);
            verts.emplace_back(p1, nor);
            verts.emplace_back(p3, nor);
            verts.emplace_back(p0, nor);
        }
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, tribuf);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(verts[0]), verts.data(), GL_DYNAMIC_DRAW);

    printf("Generated %lu vertices\n", verts.size());
}

#define EPSILON 0.00001f
uint32_t pointIndex(vector<vec3> &points, const vec3 &pt) {
    uint32_t n = points.size();
    for (uint32_t c = 0; c < n; c++) {
        if (l1Norm(pt, points[c]) < EPSILON) {
            return c;
        }
    }
    points.push_back(pt);
    return n;
}

void setup() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // Load mesh
    vector<STLTriangle> triangles;
    if (loadBinarySTL("assets/totodile.STL", triangles, true)) {
        printf("Loaded %lu triangles\n", triangles.size());
    } else {
        printf("Load failed.\n");
        return;
    }

    // Prepare quad mesh
    vec3 min, max;
    verts.reserve(triangles.size() * 3);
    uint32_t pt = 0;
    for (STLTriangle tri : triangles) {
        // Add verts
        verts.resize(pt + 3);
        verts[pt + 0].pos = tri.verts[0];
        verts[pt + 0].nor = -tri.normal;
        verts[pt + 1].pos = tri.verts[2];
        verts[pt + 1].nor = -tri.normal;
        verts[pt + 2].pos = tri.verts[1];
        verts[pt + 2].nor = -tri.normal;

        // Add to quads
        mesh.emplace_back();
        Quad &quad = mesh.back();
        quad.e[0].pos = pointIndex(positions, tri.verts[0]);
        quad.e[0].opp = BAD_OPP;
        quad.e[1].pos = pointIndex(positions, tri.verts[2]);
        quad.e[1].opp = BAD_OPP;
        quad.e[2].pos = pointIndex(positions, tri.verts[1]);
        quad.e[2].opp = BAD_OPP;
        quad.e[3].pos = BAD_OPP;
        quad.e[3].opp = BAD_OPP;

        // Update bounds
        min = glm::min(tri.verts[0], min);
        max = glm::max(tri.verts[0], max);
        min = glm::min(tri.verts[1], min);
        max = glm::max(tri.verts[1], max);
        min = glm::min(tri.verts[2], min);
        max = glm::max(tri.verts[2], max);
        pt += 3;
    }
    isTriangles = true;
    printf("%lu triangles with %lu unique points.\n", mesh.size(), positions.size());

    // Find adjacencies for the triangles
    HalfEdge *edge = reinterpret_cast<HalfEdge *>(mesh.data());
    for (uint32_t c = 0, n = mesh.size() * 4; c < n; c += 4) {
        for (uint32_t d = c + 4; d < n; d += 4) {
            // Edge c-01
            if (edge[c+0].opp == BAD_OPP) {
                if (edge[c + 0].pos == edge[d + 1].pos &&
                    edge[c + 1].pos == edge[d + 0].pos) {
                    edge[c + 0].opp = d + 0;
                    edge[d + 0].opp = c + 0;
                } else
                if (edge[c + 0].pos == edge[d + 2].pos &&
                    edge[c + 1].pos == edge[d + 1].pos) {
                    edge[c + 0].opp = d + 1;
                    edge[d + 1].opp = c + 0;
                } else
                if (edge[c + 0].pos == edge[d + 0].pos &&
                    edge[c + 1].pos == edge[d + 2].pos) {
                    edge[c + 0].opp = d + 2;
                    edge[d + 2].opp = c + 0;
                }
            }
            // Edge c-12
            if (edge[c+1].opp == BAD_OPP) {
                if (edge[c + 1].pos == edge[d + 1].pos &&
                    edge[c + 2].pos == edge[d + 0].pos) {
                    edge[c + 1].opp = d + 0;
                    edge[d + 0].opp = c + 1;
                } else
                if (edge[c + 1].pos == edge[d + 2].pos &&
                    edge[c + 2].pos == edge[d + 1].pos) {
                    edge[c + 1].opp = d + 1;
                    edge[d + 1].opp = c + 1;
                } else
                if (edge[c + 1].pos == edge[d + 0].pos &&
                    edge[c + 2].pos == edge[d + 2].pos) {
                    edge[c + 1].opp = d + 2;
                    edge[d + 2].opp = c + 1;
                }
            }
            // Edge c-20
            if (edge[c+2].opp == BAD_OPP) {
                if (edge[c + 2].pos == edge[d + 1].pos &&
                    edge[c + 0].pos == edge[d + 0].pos) {
                    edge[c + 2].opp = d + 0;
                    edge[d + 0].opp = c + 2;
                } else
                if (edge[c + 2].pos == edge[d + 2].pos &&
                    edge[c + 0].pos == edge[d + 1].pos) {
                    edge[c + 2].opp = d + 1;
                    edge[d + 1].opp = c + 2;
                } else
                if (edge[c + 2].pos == edge[d + 0].pos &&
                    edge[c + 0].pos == edge[d + 2].pos) {
                    edge[c + 2].opp = d + 2;
                    edge[d + 2].opp = c + 2;
                }
            }
        }
        if (edge[c+0].opp == BAD_OPP ||
            edge[c+1].opp == BAD_OPP ||
            edge[c+2].opp == BAD_OPP) {
            printf("Missing opposite for triangle %d\n", c / 4);
        }
    }

    // Set up surface buffer
    shader = compileShader(vert, frag);
    uniforms.mvp = glGetUniformLocation(shader, "mvp");
    uniforms.normalMat = glGetUniformLocation(shader, "normalMat");
    checkError();

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    checkError();

    glGenBuffers(1, &tribuf);
    glBindBuffer(GL_ARRAY_BUFFER, tribuf);
//    glGenBuffers(1, &trielm);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, trielm);
    checkError();

    GLuint pos = glGetAttribLocation(shader, "position");
    GLuint nor = glGetAttribLocation(shader, "normal");
    glEnableVertexAttribArray(pos);
    glEnableVertexAttribArray(nor);
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    glVertexAttribPointer(nor, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) sizeof(vec3));
    checkError();

    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

    // Set up adjacency buffer
    lineShader = compileShader(linevert, frag);
    lineUniforms.mvp = glGetUniformLocation(lineShader, "mvp");
    checkError();

    glGenVertexArrays(1, &linevao);
    glBindVertexArray(linevao);
    checkError();

    glGenBuffers(1, &linebuf);
    glBindBuffer(GL_ARRAY_BUFFER, linebuf);
    glGenBuffers(1, &lineelm);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineelm);
    checkError();

    pos = glGetAttribLocation(lineShader, "position");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), 0);
    checkError();

    // Set up transforms
    vec3 center = (min + max) / 2.f;
    float radius = length(max - min) / 2.f;

    view = lookAt(vec3(0, 0, 5), vec3(0), vec3(0, 1, 0));
    rotation = glm::scale(vec3(3/radius)) * glm::translate(-center);
    rotation = lookAt(vec3(0), vec3(-1), vec3(0, 1, 0)) * rotation;

    generateLineBuffer();
}

uint32_t nextEdge(HalfEdge *edges, uint32_t edge) {
    edge++;
    if ((edge & 3) == 0) return edge - 4;
    if (edges[edge].pos == BAD_OPP) return edge - 3;
    else return edge;
}

void subdivideMesh() {
    // First, calculate all of the face points as the average of the points of the face
    uint32_t pointSize = positions.size();
    uint32_t faceSize = mesh.size();
    uint32_t edgeSize = 2*faceSize;
    uint32_t faceBase = pointSize;
    uint32_t edgeBase = faceBase + faceSize;
    uint32_t totalSize = edgeBase + edgeSize;

    vector<uint8_t> newCts(pointSize, 0);
    vector<vec3> newPts;
    newPts.reserve(totalSize);
    newPts.resize(pointSize, vec3(0));

    // Calculate the face points
    for (uint32_t c = 0; c < faceSize; c++) {
        uint32_t p0 = mesh[c].e[0].pos;
        uint32_t p1 = mesh[c].e[1].pos;
        uint32_t p2 = mesh[c].e[2].pos;
        uint32_t p3 = mesh[c].e[3].pos;
        vec3 pt = positions[p0] +
                  positions[p1] +
                  positions[p2];
        if (p3 != BAD_OPP) {
            pt = (pt + positions[p3]) / 4.f;
        } else {
            pt /= 3.f;
        }
        newPts.push_back(pt);
        newPts[p0] += pt;
        newPts[p1] += pt;
        newPts[p2] += pt;
        newCts[p0] ++;
        newCts[p1] ++;
        newCts[p2] ++;
        if (p3 != BAD_OPP) {
            newPts[p3] += pt;
            newCts[p3] ++;
        }
    }

    assert(newPts.size() == edgeBase);

    // Calculate the edge points
    vector<uint32_t> edgePtPtrs;
    edgePtPtrs.resize(faceSize * 4, BAD_OPP);
    HalfEdge *edge = reinterpret_cast<HalfEdge *>(mesh.data());
    for (uint32_t c = 0, n = faceSize * 4; c < n; c++) {
        if (edge[c].opp != BAD_OPP && edge[c].opp > c) {
            uint32_t next = nextEdge(edge, c);
            vec3 pos = (positions[edge[c].pos] + positions[edge[next].pos] +
                       newPts[faceBase + c/4] + newPts[faceBase + edge[c].opp / 4]) / 4.f;
            uint32_t edgePtr = newPts.size();
            newPts.push_back(pos);
            edgePtPtrs[c] = edgePtr;
            edgePtPtrs[edge[c].opp] = edgePtr;

            newPts[edge[c].pos] += 2.f * pos;
            newPts[edge[next].pos] += 2.f * pos;
        }
    }

    // Calculate the updated original points
    for (uint32_t c = 0, n = pointSize; c < n; c++) {
        float k = float(newCts[c]);
        assert(k >= 3);
        newPts[c] = (newPts[c] / k + (k - 3)*positions[c]) / k;
    }

    // Calculate the new topology
    vector<Quad> newMesh;
    newMesh.reserve(4 * faceSize);
    for (uint32_t c = 0, n = faceSize; c < n; c++) {
        Quad &q = mesh[c];
        if (isTriangles) {
            // Create three quads
            uint32_t center = faceBase + c;
            uint32_t p0 = q.e[0].pos;
            uint32_t e0 = edgePtPtrs[c*4 + 0];
            uint32_t p1 = q.e[1].pos;
            uint32_t e1 = edgePtPtrs[c*4 + 1];
            uint32_t p2 = q.e[2].pos;
            uint32_t e2 = edgePtPtrs[c*4 + 2];

            uint32_t base = c * 3;

            uint32_t base01 = q.e[0].opp / 4 * 3;
            uint32_t base12 = q.e[1].opp / 4 * 3;
            uint32_t base20 = q.e[2].opp / 4 * 3;

            uint32_t off01 = q.e[0].opp & 3;
            uint32_t off12 = q.e[1].opp & 3;
            uint32_t off20 = q.e[2].opp & 3;
            uint32_t nxt01 = off01 == 2 ? 0 : off01+1;
            uint32_t nxt12 = off12 == 2 ? 0 : off12+1;
            uint32_t nxt20 = off20 == 2 ? 0 : off20+1;

            Quad q0, q1, q2;
            q0.e[0].pos = center;
            q0.e[0].opp = (base + 2)*4 + 3;
            q0.e[1].pos = e2;
            q0.e[1].opp = (base20 + off20)*4 + 2;
            q0.e[2].pos = p0;
            q0.e[2].opp = (base01 + nxt01)*4 + 1;
            q0.e[3].pos = e0;
            q0.e[3].opp = (base + 1)*4 + 0;

            q1.e[0].pos = center;
            q1.e[0].opp = (base + 0)*4 + 3;
            q1.e[1].pos = e0;
            q1.e[1].opp = (base01 + off01)*4 + 2;
            q1.e[2].pos = p1;
            q1.e[2].opp = (base12 + nxt12)*4 + 1;
            q1.e[3].pos = e1;
            q1.e[3].opp = (base + 2)*4 + 0;

            q2.e[0].pos = center;
            q2.e[0].opp = (base + 1)*4 + 3;
            q2.e[1].pos = e1;
            q2.e[1].opp = (base12 + off12)*4 + 2;
            q2.e[2].pos = p2;
            q2.e[2].opp = (base20 + nxt20)*4 + 1;
            q2.e[3].pos = e2;
            q2.e[3].opp = (base + 0)*4 + 0;

            newMesh.push_back(q0);
            newMesh.push_back(q1);
            newMesh.push_back(q2);
        } else {
            // Create four quads
            uint32_t center = faceBase + c;
            uint32_t p0 = q.e[0].pos;
            uint32_t e0 = edgePtPtrs[c*4 + 0];
            uint32_t p1 = q.e[1].pos;
            uint32_t e1 = edgePtPtrs[c*4 + 1];
            uint32_t p2 = q.e[2].pos;
            uint32_t e2 = edgePtPtrs[c*4 + 2];
            uint32_t p3 = q.e[3].pos;
            uint32_t e3 = edgePtPtrs[c*4 + 3];

            uint32_t base = c * 4;

            uint32_t base01 = q.e[0].opp & ~3;
            uint32_t base12 = q.e[1].opp & ~3;
            uint32_t base23 = q.e[2].opp & ~3;
            uint32_t base30 = q.e[3].opp & ~3;

            uint32_t off01 = q.e[0].opp & 3;
            uint32_t off12 = q.e[1].opp & 3;
            uint32_t off23 = q.e[2].opp & 3;
            uint32_t off30 = q.e[3].opp & 3;
            uint32_t nxt01 = off01 == 3 ? 0 : off01+1;
            uint32_t nxt12 = off12 == 3 ? 0 : off12+1;
            uint32_t nxt23 = off23 == 3 ? 0 : off23+1;
            uint32_t nxt30 = off30 == 3 ? 0 : off30+1;

            Quad q0, q1, q2, q3;
            q0.e[0].pos = center;
            q0.e[0].opp = (base + 3)*4 + 3;
            q0.e[1].pos = e3;
            q0.e[1].opp = (base30 + off30)*4 + 2;
            q0.e[2].pos = p0;
            q0.e[2].opp = (base01 + nxt01)*4 + 1;
            q0.e[3].pos = e0;
            q0.e[3].opp = (base + 1)*4 + 0;

            q1.e[0].pos = center;
            q1.e[0].opp = (base + 0)*4 + 3;
            q1.e[1].pos = e0;
            q1.e[1].opp = (base01 + off01)*4 + 2;
            q1.e[2].pos = p1;
            q1.e[2].opp = (base12 + nxt12)*4 + 1;
            q1.e[3].pos = e1;
            q1.e[3].opp = (base + 2)*4 + 0;

            q2.e[0].pos = center;
            q2.e[0].opp = (base + 1)*4 + 3;
            q2.e[1].pos = e1;
            q2.e[1].opp = (base12 + off12)*4 + 2;
            q2.e[2].pos = p2;
            q2.e[2].opp = (base23 + nxt23)*4 + 1;
            q2.e[3].pos = e2;
            q2.e[3].opp = (base + 3)*4 + 0;

            q3.e[0].pos = center;
            q3.e[0].opp = (base + 2)*4 + 3;
            q3.e[1].pos = e2;
            q3.e[1].opp = (base23 + off23)*4 + 2;
            q3.e[2].pos = p3;
            q3.e[2].opp = (base30 + nxt30)*4 + 1;
            q3.e[3].pos = e3;
            q3.e[3].opp = (base + 0)*4 + 0;

            newMesh.push_back(q0);
            newMesh.push_back(q1);
            newMesh.push_back(q2);
            newMesh.push_back(q3);
        }
    }

    mesh = move(newMesh);
    positions = move(newPts);
    isTriangles = false;
}

void draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glUseProgram(shader);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, verts.size());

    if (drawAdjacency) {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(lineShader);
        glBindVertexArray(linevao);
        glDrawElements(GL_LINES, linesSize, GL_UNSIGNED_INT, nullptr);
    }
}

void updateMatrices() {
    glUseProgram(shader);
    mat4 mvp = projection * view * rotation;
    glUniformMatrix4fv(uniforms.mvp, 1, GL_FALSE, &mvp[0][0]);
    mat3 normal = mat3(view * rotation);
    glUniformMatrix3fv(uniforms.normalMat, 1, GL_FALSE, &normal[0][0]);

    glUseProgram(lineShader);
    glUniformMatrix4fv(lineUniforms.mvp, 1, GL_FALSE, &mvp[0][0]);
}

static void glfw_resize_callback(GLFWwindow *window, int width, int height) {
    printf("resize: %dx%d\n", width, height);
    glViewport(0, 0, width, height);
    if (height != 0) {
        float aspect = float(width) / height;
        projection = perspective(62.f, aspect, 0.5f, 10.f);
        updateMatrices();
    }
}

static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, true);
    } else if (key == GLFW_KEY_W) {
        static bool wireframe = false;
        wireframe = !wireframe;
        glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    } else if (key == GLFW_KEY_S) {
        subdivideMesh();
        generateLineBuffer();
        generateMeshBuffer();
    } else if (key == GLFW_KEY_A) {
        drawAdjacency = !drawAdjacency;
    }
}

vec2 lastMouse = vec2(-1,-1);

static void glfw_click_callback(GLFWwindow *window, int button, int action, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) lastMouse = vec2(x, y);
        else if (action == GLFW_RELEASE) lastMouse = vec2(-1, -1);
    }
    checkError();
}

static void glfw_mouse_callback(GLFWwindow *window, double xPos, double yPos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) return;
    if (lastMouse == vec2(-1,-1)) {
        lastMouse = vec2(xPos, yPos);
        return; // can't update this frame, no previous data.
    } else {
        vec2 current = vec2(xPos, yPos);
        vec2 delta = current - lastMouse;
        if (delta == vec2(0,0)) return;

        vec3 rotationVector = vec3(delta.y, delta.x, 0);
        float angle = length(delta);
        rotation = rotate(angle, rotationVector) * rotation;
        updateMatrices();

        lastMouse = current;
    }
    checkError();
}

void glfw_error_callback(int error, const char* description) {
    cerr << "GLFW Error: " << description << " (error " << error << ")" << endl;
    checkError();
}

void checkShaderError(GLuint shader) {
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success) return;

    cout << "Shader Compile Failed." << endl;

    GLint logSize = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
    if (logSize == 0) {
        cout << "No log found." << endl;
        return;
    }

    GLchar *log = new GLchar[logSize];

    glGetShaderInfoLog(shader, logSize, &logSize, log);

    cout << log << endl;

    delete[] log;
}

void checkLinkError(GLuint program) {
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success) return;

    cout << "Shader link failed." << endl;

    GLint logSize = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
    if (logSize == 0) {
        cout << "No log found." << endl;
        return;
    }

    GLchar *log = new GLchar[logSize];

    glGetProgramInfoLog(program, logSize, &logSize, log);
    cout << log << endl;

    delete[] log;
}

GLuint compileShader(const char *vertSrc, const char *fragSrc) {
    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertSrc, nullptr);
    glCompileShader(vertex);
    checkShaderError(vertex);

    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragSrc, nullptr);
    glCompileShader(fragment);
    checkShaderError(fragment);

    GLuint shader = glCreateProgram();
    glAttachShader(shader, vertex);
    glAttachShader(shader, fragment);
    glLinkProgram(shader);
    checkLinkError(shader);

    return shader;
}

void loadTexture(GLuint texname, const char *filename) {
    glBindTexture(GL_TEXTURE_2D, texname);

    int width, height, bpp;
    unsigned char *pixels = stbi_load(filename, &width, &height, &bpp, STBI_default);
    if (pixels == nullptr) {
        cout << "Failed to load image " << filename << " (" << stbi_failure_reason() << ")" << endl;
        return;
    }
    cout << "Loaded " << filename << ", " << height << 'x' << width << ", comp = " << bpp << endl;

    GLenum format;
    switch(bpp) {
        case STBI_rgb:
            format = GL_RGB;
            break;
        case STBI_rgb_alpha:
            format = GL_RGBA;
            break;
        default:
            cout << "Unsupported format: " << bpp << endl;
            return;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);
}

int main() {
    if (!glfwInit()) {
        cout << "Failed to init GLFW" << endl;
        exit(-1);
    }
    cout << "GLFW Successfully Started" << endl;

    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#ifdef APPLE
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    window = glfwCreateWindow(640, 480, "SpexGuy's GLFW Template", NULL, NULL);
    if (!window) {
        cout << "Failed to create window" << endl;
        exit(-1);
    }

    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetMouseButtonCallback(window, glfw_click_callback);
    glfwSetCursorPosCallback(window, glfw_mouse_callback);

    glfwMakeContextCurrent(window);

    // If the program is crashing at glGenVertexArrays, try uncommenting this line.
    //glewExperimental = GL_TRUE;
    glewInit();

    printf("OpenGL version recieved: %s\n", glGetString(GL_VERSION));

    glfwSwapInterval(1);

    initPerformanceData();

    setup();
    checkError();

    glfwSetFramebufferSizeCallback(window, glfw_resize_callback); // do this after setup
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfw_resize_callback(window, width, height); // call resize once with the initial size

    // make sure performance data is clean going into main loop
    markPerformanceFrame();
    printPerformanceData();
    double lastPerfPrintTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {

        {
            Perf stat("Poll events");
            glfwPollEvents();
            checkError();
        }
        {
            Perf stat("Draw");
            draw();
            checkError();
        }
        {
            Perf stat("Swap buffers");
            glfwSwapBuffers(window);
            checkError();
        }

        markPerformanceFrame();

        double now = glfwGetTime();
        if (now - lastPerfPrintTime > 10.0) {
            printPerformanceData();
            lastPerfPrintTime = now;
        }
    }

    return 0;
}
