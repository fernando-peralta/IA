//#include "graph.h"
//#include "graficos.h"
#include "geneticAlgorithm.h"

#define WINDOW_WIDTH  600
#define WINDOW_HEIGHT 600
// ------------------------------------------- Parte principal -------------------------------------------

int windowWidth = 1000, windowHeight = 1000;

int once  = 0;
int iniNode = -10;
int endNode = -10;
int selection = 0;
int active_search = 0;
GLuint textureID; // Variable para almacenar el ID de la textura


void addCircles(){
    circles.clear();
    int counter = 0;
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
                if(initiated &&  graph.coord_to_node(i,j) == generation[0][0]) {
                    //if(once < 2) printf("\n Node 1?? : %d", graph.coord_to_node(i,j));
                    circle.color[0] = 1.0, circle.color[1] = 0.0, circle.color[2] = 0.0;
                    //once++;
                }
                else if(initiated &&  graph.coord_to_node(i,j) == generation[0][generation[0].size()-2]) {
                    //if(once < 2) printf("\n Node 2?? : %d\n", graph.coord_to_node(i,j));
                    circle.color[0] = 0.5, circle.color[1] = 0.5, circle.color[2] = 0.75;
                    //once++;
                }
                else circle.color[0] = 0, circle.color[1] = 0, circle.color[2] = 1;
               
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
    cout << "1. Show a certain amount of nodes " << endl;
    cout << "2. Original screen" << endl;
    cout << "3. Run genetic algorithm again " << endl;
    cout << "4. Exit" << endl;
    
}



// Función que se llama cuando se dibuja la ventana
void display(){
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Nodes drawing
    for(int i = 0; i < circles.size(); ++i) drawCircle(circles[i]);

    for(int i = 0; i < lines.size(); ++i) drawLine(lines[i]);

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
        // Se hizo clic con el botón izquierdo, estado es hacia abajo
        selection = (selection + 1)%2;
        x -= windowWidth / 2;
        y = windowHeight / 2 - y;
        printf("Clic en la posición: (%d, %d)\n", x, y);
        for(auto& circle: circles){
            //cout << "distance: " << sqrt(pow(x - abs(circle.x),2) + pow(y-abs(circle.y),2)) << endl;
            if (sqrt(pow(x - circle.x,2) + pow(y-circle.y,2)) <= circle.radius){
                if(selection){ iniNode = circle.number; cout << "ini_node: " << iniNode << endl;}
                else {endNode = circle.number; cout << "end_node: " << endNode << endl;}
                circle.color[0] = circle.color[1] = circle.color[2] = 0.3;
                return;
            }   
        }
    }
}


void printPath(vector<int>& path){
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
    if(key == '4'){
        glutDestroyWindow(glutGetWindow());
    }
    else{
        if(key == '2'){
            graph = Graph(graphSize, graphSize);
            addCircles();
            addLines();
        }
        else if(key == '1'){
            cout << "Number of nodes to show: ";
            float num;
            cin >> num;
            graph.keepRandomNodes(num);
            printNodes();
            genetic_algorithm();
            circles.clear();
            lines.clear();
            addCircles();
            addLines();
            //active_search = 0;
        }

        else if(key == '3'){
            printNodes();
            genetic_algorithm();
            circles.clear();
            lines.clear();
            addCircles();
            addLines();
            //active_search = 0;
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
