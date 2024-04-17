cmake_minimum_required(VERSION 3.25)
project(Coloreo_Grafos)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(Coloreo_Grafos main.cpp)

target_link_libraries(Coloreo_Grafos -lOpenGL32 -lfreeGLUT -lglu32)
target_link_libraries(Coloreo_Grafos ${OPENGL_LIBRARIES})
target_link_libraries(Coloreo_Grafos ${GULT_LIBRARIES})