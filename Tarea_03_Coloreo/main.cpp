#include <iostream>
#include <GL/glut.h>
#include <cmath>
#include <vector>

using namespace std;

#define M_PI 3.14159265358979323846

// Estructura de un círculo con color
struct Circle {
    float x;
    float y;
    float radius;
    float color[3];
    int number;
};

// Estructura de una línea con color
struct Line {
    float x1;
    float y1;
    float x2;
    float y2;
    float color[3];
    float width;
};

// Estructura para representar el grafo
struct Graph {
    int V;
    vector<vector<int>> adj;
};

vector<pair<float, float>> nodes = { {-0.5, 0.0}, {-0.5, -0.5}, {0, -0.5}, {0, 0.0}, {0.5, -0.25} };
vector<pair<int, int>> edges = { {1, 2}, {1, 4}, {1, 5}, {2, 3}, {3, 4}, {3, 5}, {4, 5} };

vector<pair<float, float>> nodes2 = { {-0.8, 0.2}, {-0.8, -0.2}, {-0.2, -0.2}, {-0.2, 0.2}, {0.2, 0.2}, {0.2, -0.2}, {0.8, 0.0}};
vector<pair<int, int>> edges2 = { {1, 2}, {1, 4}, {2, 3}, {2, 7}, {3, 4},{3,6}, {4, 5}, {4, 6}, {5, 7},{5,6}, {6, 7}};


// Función para inicializar el grafo
Graph initGraph(int V) {
    Graph G;
    G.V = V;
    G.adj.resize(V);
    return G;
}

// Función para agregar una arista al grafo
void addEdge(Graph& G, int u, int v) {
    G.adj[u].push_back(v);
    G.adj[v].push_back(u);
}

Graph G;  // Grafo global
vector<int> color;      // Colores asignados usando heurística de grado máximo
vector<int> color2;     // Colores asignados usando heurística de variable más restringida
int Cantidad_Colores;   // Cantidad de colores disponibles
int Nodo_Inicio;        // Nodo de inicio para la heurística de variable más restringida

// Dibuja un círculo
void drawCircle(Circle circle) {
    glColor3f(circle.color[0], circle.color[1], circle.color[2]);
    glBegin(GL_POLYGON);
    for (int i = 0; i < 360; ++i) {
        float degInRad = i * M_PI / 180;
        glVertex2f(circle.x + cos(degInRad) * circle.radius, circle.y + sin(degInRad) * circle.radius);
    }
    glEnd();
}

// Dibuja una línea
void drawLine(Line line) {
    glColor3f(line.color[0], line.color[1], line.color[2]);
    glLineWidth(line.width);
    glBegin(GL_LINES);
    glVertex2f(line.x1, line.y1);
    glVertex2f(line.x2, line.y2);
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT); // Limpia la pantalla
    glLoadIdentity(); // Restablece transformaciones

    // Dibujar aristas
    for (auto& edge : edges) {
        int u = edge.first - 1;
        int v = edge.second - 1;
        if (u >= 0 && u < G.V && v >= 0 && v < G.V) {
            // Obtener las coordenadas de los nodos
            float x1 = nodes[u].first;
            float y1 = nodes[u].second;
            float x2 = nodes[v].first;
            float y2 = nodes[v].second;

            // Dibujar línea entre los nodos
            Line line = { x1, y1, x2, y2, {1.0, 1.0, 1.0}, 2.0 };
            drawLine(line);
        }
        else {
            cout << "Índice fuera de rango al dibujar arista: " << u << ", " << v << endl;
        }
    }

    // Dibujar nodos con colores asignados
    for (size_t i = 0; i < nodes.size(); ++i) {
        // Obtener las coordenadas del nodo
        float x = nodes[i].first;
        float y = nodes[i].second;
        int col;
        if(color[0]!=-1){
            col = color[i];
        }else{
            col=color2[i];
        }

        // Obtener el color asignado al nodo

        float r, g, b;
        if (col == 0) {
            r = 1.0;
            g = 0.0;
            b = 0.0; // Rojo
        } else if (col == 1) {
            r = 0.0;
            g = 1.0;
            b = 0.0; // Verde
        } else if (col == 2) {
            r = 0.0;
            g = 0.0;
            b = 1.0; // Azul
        }else if (col == 3) {
            r = 0.0;
            g = 0.5;
            b = 1.0; // Celeste
        }else {
            // Por defecto, se usa blanco si el color no está definido
            r = 1.0;
            g = 1.0;
            b = 1.0;
        }

        // Dibujar círculo en la posición del nodo con el color correspondiente
        Circle circle = { x, y, 0.05, {r, g, b}, 1 };
        drawCircle(circle);
    }

    glFlush(); // Procesa todas las rutinas de OpenGL lo más rápido posible
}

void display2() {
    glClear(GL_COLOR_BUFFER_BIT); // Limpia la pantalla
    glLoadIdentity(); // Restablece transformaciones

    // Dibujar aristas
    for (auto& edge : edges2) {
        int u = edge.first - 1;
        int v = edge.second - 1;
        if (u >= 0 && u < G.V && v >= 0 && v < G.V) {
            // Obtener las coordenadas de los nodos
            float x1 = nodes2[u].first;
            float y1 = nodes2[u].second;
            float x2 = nodes2[v].first;
            float y2 = nodes2[v].second;

            // Dibujar línea entre los nodos
            Line line = { x1, y1, x2, y2, {1.0, 1.0, 1.0}, 2.0 };
            drawLine(line);
        }
        else {
            cout << "Indice fuera de rango al dibujar arista: " << u << ", " << v << endl;
        }
    }

    // Dibujar nodos con colores asignados
    for (size_t i = 0; i < nodes2.size(); ++i) {
        // Obtener las coordenadas del nodo
        float x = nodes2[i].first;
        float y = nodes2[i].second;

        // Obtener el color asignado al nodo
        int col;
        if(color[0]!=-1){
            col = color[i];
        }else{
            col=color2[i];
        }
        float r, g, b;
        if (col == 0) {
            r = 1.0;
            g = 0.0;
            b = 0.0; // Rojo
        } else if (col == 1) {
            r = 0.0;
            g = 1.0;
            b = 0.0; // Verde
        } else if (col == 2) {
            r = 0.0;
            g = 0.0;
            b = 1.0; // Azul
        }
        else if (col == 3) {
            r = 0.0;
            g = 0.5;
            b = 1.0; // Celeste
        }else {
            // Por defecto, se usa blanco si el color no está definido
            r = 1.0;
            g = 1.0;
            b = 1.0;
        }

        // Dibujar círculo en la posición del nodo con el color correspondiente
        Circle circle = { x, y, 0.05, {r, g, b}, 1 };
        drawCircle(circle);
    }

    glFlush(); // Procesa todas las rutinas de OpenGL lo más rápido posible
}





//Comprobamos que es seguro asignar un color a un nodo sin violar las restricciones
bool isSafe(Graph& G, int v, vector<int> color, int c) {
    for (int u : G.adj[v]) {
        if (color[u] == c) return false;
    }
    return true;
}

// Función para colorear el grafo utilizando la heurística de grado máximo
bool graphColoring_Grado_Maximo(Graph& G, int Cantidad_Colores, vector<int>& color, int v = 0) {
    if (v == G.V) // Si todos los vértices están coloreados, termina
        return true;

    // Encuentra el nodo con más conexiones no coloreado
    int max_degree_node = 1;
    int max_degree = -1;

    for (int i = 0; i < G.V; ++i) {
        if (color[i] == -1) {
            int auxiliar_depu = G.adj[i].size();
            if (auxiliar_depu > max_degree) {
                max_degree_node = i;
                max_degree = G.adj[i].size();
            }
        }
    }

    // Intentar diferentes colores para el vértice seleccionado
    for (int c = 0; c < Cantidad_Colores; c++) {
        if (isSafe(G, max_degree_node, color, c)) {
            color[max_degree_node] = c; // Asignar el color

            // Imprimir el coloreo actual
            cout << "Nodo " << max_degree_node + 1 << " -> Color " << color[max_degree_node] + 1 << endl;

            if (graphColoring_Grado_Maximo(G, Cantidad_Colores, color, v + 1))
                return true;

            color[max_degree_node] = -1; // Revertir si no funciona
        }
    }

    return false;
}

bool graphColoring_Restringida(Graph& G, int Cantidad_Colores, vector<int>& color, int v=1) {
    if (v == G.V) // Si todos los vértices están coloreados, termina
        return true;

    // Encontrar el próximo nodo no coloreado con el mayor número de nodos adyacentes coloreados
    int max_colored_adjacent_node = -1;
    int max_colored_adjacent = -1;

    for (int i = 0; i < G.V; ++i) {
        if (color[i] == -1) {
            int num_colored_adjacent = 0;
            for (int j : G.adj[i]) {
                if (color[j] != -1) {
                    num_colored_adjacent++;
                }
            }
            if (num_colored_adjacent > max_colored_adjacent) {
                max_colored_adjacent_node = i;
                max_colored_adjacent = num_colored_adjacent;
            }
        }
    }

    // Si no hay nodos disponibles para colorear, devolver falso
    if (max_colored_adjacent_node == -1)
        return false;

    // Contar la cantidad de colores disponibles para el nodo
    for (int c = 0; c < Cantidad_Colores; ++c) {
        if (isSafe(G, max_colored_adjacent_node, color, c)) {
            color[max_colored_adjacent_node] = c; // Asignar el color

            // Imprimir el coloreo actual
            cout << "Nodo " << max_colored_adjacent_node + 1 << " -> Color " << color[max_colored_adjacent_node] + 1 << endl;

            // Llamada recursiva con el siguiente nodo
            if (graphColoring_Restringida(G, Cantidad_Colores, color, v + 1))
                return true;

            color[max_colored_adjacent_node] = -1; // Revertir si no funciona
        }
    }

    return false; // Si no se encontró una solución válida
}



void keyboard(unsigned char key, int x, int y) {
    bool Nodo_Valido = false;
    switch (key)
    {
        case '1':
            if (graphColoring_Grado_Maximo(G, Cantidad_Colores, color)) {
                cout << "Solucion encontrada con Grado Maximo:" << endl;
                for (int u = 0; u < G.V; ++u) {
                    cout << "Nodo " << u + 1 << " -> Color " << color[u] + 1 << endl;
                }
            }
            else {
                cout << "No se encontró solución con " << Cantidad_Colores << " colores." << endl;
            }
            break;
        case '2':

            /*while (!Nodo_Valido)
            {
                cout << "Con qué nodo desea empezar? (Entre 0 y " << G.V-1 << "): "; cin >> Nodo_Inicio;
                if (Nodo_Inicio < 0 || Nodo_Inicio > G.V-1)
                {
                    cout << "Seleccione un nodo entre 0 y " << G.V << endl;
                }
                else
                {
                    Nodo_Valido = true;
                }
            }*/
            color2[2] = 0;
            if (graphColoring_Restringida(G, Cantidad_Colores, color2)) {
                cout << "Solucion encontrada variable mas restringida:" << endl;
                for (int u = 0; u < G.V; ++u) {
                    cout << "Nodo " << u + 1 << " -> Color " << color2[u] + 1 << endl;
                }
            }
            else {
                cout << "No se encontro solución con " << Cantidad_Colores << " colores." << endl;
            }
            break;
        case '3':
            for (int i = 0; i < color.size(); ++i) {
                color[i]=-1;
                color2[i]=-1;
            }

        case '4':
            exit(0);
            break;

        default:
            cout << "Opcion no válida." << endl;
            break;
    }
    glutPostRedisplay(); // Pide a GLUT redibujar la ventana
}

int main(int argc, char** argv)  {
    // Inicialización de GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Visualizacion de Grafo");

    // Configura el espacio de dibujo
    glClearColor(0.0, 0.0, 0.0, 1.0); // Fondo negro
    glColor3f(1.0, 1.0, 1.0); // Color blanco para dibujar
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);



    // Inicializar variables globales
    vector<char> letras = { 'A', 'B', 'C', 'D', 'E' };
    int grafo=2;
    if  (grafo==1) {
        G = initGraph(5);
        // Agregar aristas al grafo
        for (auto& edge : edges) {
            int u = edge.first - 1;
            int v = edge.second - 1;
            if (u >= 0 && u < G.V && v >= 0 && v < G.V) {
                addEdge(G, u, v);
            }
            else {
                cout << "Índice fuera de rango al añadir arista: " << u << ", " << v << endl;
            }
        }
        glutDisplayFunc(display);

        // Registra funciones de callback

        glutKeyboardFunc(keyboard);

        // Inicializar colores
        color.resize(G.V, -1);
        color2.resize(G.V, -1);
        Cantidad_Colores = 3;

    }else{
        G = initGraph(7);
        // Agregar aristas al grafo
        for (auto& edge : edges2) {
            int u = edge.first - 1;
            int v = edge.second - 1;
            if (u >= 0 && u < G.V && v >= 0 && v < G.V) {
                addEdge(G, u, v);
            }
            else {
                cout << "Índice fuera de rango al anadir arista: " << u << ", " << v << endl;
            }
        }
        glutDisplayFunc(display2);

        // Registra funciones de callback

        glutKeyboardFunc(keyboard);

        // Inicializar colores
        color.resize(G.V, -1);
        color2.resize(G.V, -1);
        Cantidad_Colores = 4;
    }




    glutMainLoop();
    return 0;
}
