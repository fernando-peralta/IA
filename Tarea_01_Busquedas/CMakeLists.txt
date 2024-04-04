cmake_minimum_required(VERSION 3.25)
project(Tarea_01)

set(CMAKE_CXX_STANDARD 17)

add_executable(Tarea_01 main.cpp graficos.h graph.h)

target_link_libraries(Tarea_01 -lOpenGL32 -lfreeGLUT -lglu32)
target_link_libraries(Tarea_01 ${OPENGL_LIBRARIES})
target_link_libraries(Tarea_01 ${GLUT_LIBRARIES})
