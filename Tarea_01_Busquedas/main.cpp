#include "graph.h"
#include "graficos.h"

#define WINDOW_WIDTH  600
#define WINDOW_HEIGHT 600
// ------------------------------------------- Parte principal -------------------------------------------
int graphSize = 20;
int windowWidth = 600, windowHeight = 600;

// Creamos el grafo
Graph graph(graphSize, graphSize);

vector<Line> lines;
vector<Circle> circles;
vector<Line> foundPath;


int iniNode = -10;
int endNode = -10;
int selection = 0;
int active_search = 0;


void addCircles(){
    circles.clear();

    float radius = min(windowWidth / graph.cols, windowHeight / graph.rows) / 4.5;
    float divisionW = (windowWidth*0.95/graphSize);
    float divisionH = windowHeight*0.95/graphSize;
    for(int i = 0; i < graphSize; ++i){
        for(int j = 0; j < graphSize; ++j){
            if(graph.nodes[i][j]){
                Circle circle{};
                circle.x = -(windowWidth / 2) + (windowWidth*0.05) + (i) * divisionW;
                circle.y = windowHeight/2 - (windowWidth*0.05) - (j)*divisionH;
                circle.radius = radius;
                circle.number = i * graphSize + j;
                circle.color[0] = 0; circle.color[1] = 0; circle.color[2] = 1;
                circles.push_back(circle);
            }
        }
    }
}

// Funcion que halle el indicie de un elemento de un vector
int findIndex(int element){
    vector<int>v = graph.getNodes();
    for(int i = 0; i < v.size(); ++i){
        if(v[i] == element){
            return i;
        }
    }
    return -1;
}


// Funcion que agrega las conexiones entre los nodos que existen en el grafo al vector de lineas
void addLines(){
    // Se limpia el vector de lineas
    lines.clear();

    // Se agregan las conexiones entre los nodos que existen en el grafo al vector de lineas
    for(int i = 0; i < graph.n_nodes; ++i){
        for(int j = 0; j < graph.n_nodes; ++j){
            if(graph.matrix[i][j]){
                Line line;
                int a = findIndex( i);
                int b = findIndex( j);
                line.x1 = circles[a].x, line.y1 = circles[a].y;
                line.x2 = circles[b].x, line.y2 = circles[b].y;
                line.color[0] = 0, line.color[1] = 0, line.color[2] = 0.8;
                lines.push_back(line);
            }

        }
    }
}

// Funcion que imprime el menu
void printMenu(){
    cout << "Click on two nodes to set a path to look for and\n";
    cout << "1. For DFS - 2. For BFS - 3.For A* - 4. For HillClimbing" << endl;
    cout << "5. Delete percentage of nodes from screen" << endl;
    cout << "6. Original screen" << endl;
    cout << "7. Exit" << endl;
    //cout << "8. Imprime los circulos: " << endl;
    
}



// Función que se llama cuando se dibuja la ventana
void display(){
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Nodes drawing
    for(int i = 0; i < circles.size(); ++i) drawCircle(circles[i]);

    for(int i = 0; i < lines.size(); ++i) drawLine(lines[i]);

    for(int i = 0; i < foundPath.size(); ++i) drawLine(foundPath[i]);

    if (foundPath.empty() && active_search) {
        glColor3f(1.0f, 0.0f, 0.0f); // Establece el color del texto en rojo

        // Dibuja un cuadrado como una caja de texto
        glBegin(GL_QUADS);
        glVertex2f(-90.0f, 25.0f);
        glVertex2f(110.0f, 25.0f);
        glVertex2f(110.0f, -15.0f);
        glVertex2f(-90.0f, -15.0f);
        glEnd();

        glColor3f(1.0f, 1.0f, 1.0f); // Establece el color del texto en blanco
        glRasterPos2f(-60.0f, 0.0f); // Establece la posición inicial para el texto

        // Renderiza cada carácter del texto "Path not found"
        for (const char *c = "Path not found"; *c != '\0'; ++c) {
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);
        }


    }
    glutSwapBuffers();
}

void on_resize(int w, int h)
{
    windowWidth = w;
    windowHeight = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-w / 2, w / 2, -h / 2, h / 2, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    circles.clear();
    addCircles();
    lines.clear();
    addLines();
    display(); // refresh window.
}



void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        selection = (selection + 1)%2;
        x -= windowWidth / 2;
        y = windowHeight / 2 - y;
        printf("Clic en la posición: (%d, %d)\n", x, y);
        for(auto& circle: circles){

            if (sqrt(pow(x - circle.x,2) + pow(y-circle.y,2)) <= circle.radius){
                if(selection){ iniNode = circle.number; cout << "ini_node: " << iniNode << endl;}
                else endNode = circle.number; //cout << "end_node: " << endNode << endl;}
                circle.color[0] = circle.color[1] = circle.color[2] = 0.3;
                return;
            }   
        }
    }
}


void printPath(vector<int>& path){
        // Se agregan las lineas que representan el camino encontrado por el algoritmo DFS con grosor 2
        if(path.size() != 0){
            for(int i = 0; i < path.size() - 1; ++i){
                Line line;
                int a = findIndex(path[i]);
                int b = findIndex(path[i + 1]);
                line.x1 = circles[a].x, line.y1 = circles[a].y;
                line.x2 = circles[b].x, line.y2 = circles[b].y;
                line.color[0] = 1, line.color[1] = 0, line.color[2] = 0;
                line.width = 2;
                foundPath.push_back(line);
            }
        }
}

// Función que se llama cuando se presiona una tecla
void keyboard(unsigned char key, int x, int y){

    vector<int>path;
    if(key == '7'){
        glutDestroyWindow(glutGetWindow());
    }
    else{
        if(key == '6'){
            graph = Graph(graphSize, graphSize);
            foundPath.clear();

            addCircles();
            addLines();
            active_search = 0;
        }
        else if(key == '5'){
            cout << "% of nodes to erase: ";
            float num;
            cin >> num;
            graph.removeNodes(num);
            circles.clear();
            lines.clear();
            addCircles();
            addLines();
            active_search = 0;
        }
        else{
            if(iniNode != -10 && endNode != -10){
                if(key=='1') path = graph.DFS(iniNode, endNode);
                else if(key=='2') path = graph.BFS(iniNode, endNode);
                else if(key=='3') path = graph.AStar(iniNode, endNode);
                else if(key=='4') path = graph.Hillclimbling(iniNode, endNode);
                active_search = 1;
                foundPath.clear();
                printPath(path);
                //Reset
                iniNode = endNode = -10;
            }
        }
        glutPostRedisplay();
        printMenu();
    }
}


int main(int argc, char **argv){

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Grafo");
    glutDisplayFunc(display);
    glutReshapeFunc(on_resize);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouseClick);

    //addCircles();
    //addLines();
    //init();
    printMenu();
    glutMainLoop();

    return 0;
}
