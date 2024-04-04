#include "graph.h"
// ------------------------------------------- Parte de OpenGL -------------------------------------------

// Estructura de un circulo con color
struct Circle{
    float x;
    float y;
    float radius;
    float color[3];
    int number;
};

// Estructura de una linea con color
struct Line{
    float x1;
    float y1;
    float x2;
    float y2;
    float color[3];
    float width=2;
};

// Dibuja un circulo
void drawCircle(Circle circle){
    glColor3f(circle.color[0], circle.color[1], circle.color[2]);
    glBegin(GL_POLYGON);
    for(int i = 0; i < 360; ++i){
        float degInRad = i * M_PI / 180;
        glVertex2f(circle.x + cos(degInRad) * circle.radius, circle.y + sin(degInRad) * circle.radius);
    }
    glEnd();
}

// Dibuja una linea
void drawLine(Line line){
    glColor3f(line.color[0], line.color[1], line.color[2]);
    glLineWidth(line.width);
    glBegin(GL_LINES);
    glVertex2f(line.x1, line.y1);
    glVertex2f(line.x2, line.y2);
    glEnd();
}