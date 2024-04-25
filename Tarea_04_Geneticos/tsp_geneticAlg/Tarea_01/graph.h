//
// Created by marck on 8/09/2023.
//

#ifndef TAREA_01_GRAPH_H
#define TAREA_01_GRAPH_H

#include <iostream>
#include <stack>
#include <queue>
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>
#include <GL/glut.h>
#include <cmath>
#include <math.h>
#include <climits>
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <limits.h>

using namespace std;

// ------------------------------------------- Parte de estructuras de datos -------------------------------------------

// Declaración de dos arreglos de enteros que representan las posiciones relativas de los vecinos de una celda en una matriz de 8 vecinos.
int X[8] = {-1,  0, +1,  0, -1, -1, +1, +1};
int Y[8] = { 0, -1,  0, +1, -1, +1, -1, +1};

// Declaración de la clase Graph
class Graph{
public:

    // Atributos de la clase
    int n_nodes, rows, cols; // Número de nodos, número de filas y número de columnas

    // Matriz de booleanos que indica si un nodo existe o no
    bool **nodes; // 1 = exist, 0 = doesnt exist (Para eliminarlos)

    // Matriz de adyacencia que indica las conexiones entre los nodos
    bool **matrix; // Adyacencia

    // Constructor de la clase
    Graph(int rows_, int cols_){

        // Inicialización de los atributos de la clase
        rows = rows_; // Número de filas
        cols = cols_; // Número de columnas
        n_nodes = rows*cols; // Número de nodos

        // Inicialización de la matriz de adyacencia
        matrix = new bool*[n_nodes]; // Se crea un arreglo de apuntadores a arreglos de booleanos
        for (int i = 0; i < n_nodes; ++i){ // Se inicializa cada arreglo de booleanos
            matrix[i] = new bool[n_nodes]; // Se crea un arreglo de booleanos
            for (int j = 0; j < n_nodes; ++j) matrix[i][j] = 0; // No connections
        }

        // Inicialización de la matriz de nodos
        nodes = new bool*[rows]; // Se crea un arreglo de apuntadores a arreglos de booleanos
        for (int i = 0; i < rows; ++i){
            nodes[i] = new bool[cols]; // Se crea un arreglo de booleanos
            for (int j = 0; j < cols; ++j) nodes[i][j] = 1; // All nodes exist
        }

        // Establecimiento de las conexiones entre los nodos
        for (int i = 0; i < rows; ++i){ // Para cada fila
            for (int j = 0; j < cols; ++j){ // Para cada columna
                for (int k = 0; k < 8; ++k){ // Para cada vecino
                    if (i + Y[k] >= 0 && i + Y[k] < rows && j + X[k] >= 0 && j + X[k] < cols){ // Si el vecino existe
                        matrix[coord_to_node(i, j)][coord_to_node(i + Y[k], j + X[k])] = 1; // Se establece la conexión entre el nodo actual y el vecino
                        matrix[coord_to_node(i + Y[k], j + X[k])][coord_to_node(i, j)] = 1; // Se establece la conexión entre el vecino y el nodo actual
                    }
                }
            }
        }

        // Impresión de la matriz de nodos y la matriz de adyacencia
        /*print_nodes();
        cout << endl << endl;

        print_matrix();
        cout << endl << endl;*/


    }

    // Método que convierte las coordenadas de una celda en la matriz en el índice correspondiente en la matriz de adyacencia
    int coord_to_node(int row_, int col_){
        return row_*cols + col_;
    }

    // Método que imprime la matriz de nodos
    void print_nodes(){
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j) cout << nodes[i][j] << " ";
            cout << endl;
        }
    }

    // Método que imprime la matriz de adyacencia
    void print_matrix(){
        for (int i = 0; i < n_nodes; ++i){
            for (int j = 0; j < n_nodes; ++j) cout << matrix[i][j] << " ";
            cout << endl;
        }
    }

    int heuristicDistance(int start, int end){
        return abs(start/cols - end/cols) + abs(start%cols - end%cols);
    }

    pair<int, int> node_to_coord(int node) {
        int row_ = node / cols;
        int col_ = node % cols;
        return make_pair(row_, col_);
    }

    float distance(int node1, int node2){
        pair<int,int> coord1 = node_to_coord(node1);
        pair<int,int> coord2 = node_to_coord(node2);

        return sqrt(pow(coord1.first - coord2.first, 2) +
                    pow(coord1.second - coord2.second, 2));
    }

    void ady(int node){
        for(int i = 0; i <n_nodes; ++i) if(matrix[node][i]) cout << i << " ";
        cout << endl;
    }
    // Método que elimina nodos aleatorios del grafo
    void removeNodes(float num){
        srand(time(nullptr));

        int nodesRemoved = (int) n_nodes * num / 100;
        int randomNode;
        vector<int> totalNodes;
        for(int i = 0; i < n_nodes; ++i){
            totalNodes.push_back(i);
        }
        random_device rd;
        mt19937 g(rd());
        shuffle(totalNodes.begin(), totalNodes.end(), g);

        //cout << "Nodes being erased: ";
        for(int i = 0; i < nodesRemoved; ++i){
            randomNode = totalNodes[i];
            //cout << randomNode << " ";


            for(int j = 0 ; j < n_nodes; ++j){
                matrix[randomNode][j] = matrix[j][randomNode] = 0;
            }
            nodes[(int) randomNode / cols][randomNode % cols] = 0;
        }
        cout << endl;
        //return totalNodes;
    }

    // Método que elimina un nodo específico del grafo
    void removeNode(int row, int col){
        int node = coord_to_node(row, col);
        for(int i = 0; i < n_nodes; ++i){
            matrix[node][i] = matrix[i][node] = 0;
        }
        nodes[row][col] = 0;
    }
    vector<int> getNodes(){
        vector<int> nnn;
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j) 
                if(nodes[i][j])  {
                    nnn.push_back(coord_to_node(i,j));
                }
        }
        return nnn;
    }

    void keepRandomNodes(int numToKeep) {
        // Obtener todos los nodos existentes
        vector<int> existingNodes = getNodes();
        /*for(auto& n : existingNodes){
            printf("%d ", n);
        }
        printf("\n");
        */
        random_shuffle(existingNodes.begin(), existingNodes.end());

        // Mantener solo los primeros 'numToKeep' nodos
        vector<int> nodesToKeep(existingNodes.begin(), existingNodes.begin() + numToKeep);
        
        //for(auto& n : nodesToKeep){
          //  printf("%d ", n);
       //}
        //printf("\n");
        // Eliminar todos los nodos restantes y sus conexiones
        for (int i = 0; i < n_nodes; ++i) {
            if (find(nodesToKeep.begin(), nodesToKeep.end(), i) == nodesToKeep.end()) {
                for (int j = 0; j < n_nodes; ++j) {
                    matrix[i][j] = matrix[j][i] = 0;
                }
                nodes[(int) (i / cols)][i % cols] = 0;
            }
        }
        reconnect();
    }

    void reconnect(){
        vector<int> existingNodes = getNodes();
        for (int i = 0; i < existingNodes.size(); ++i) {
            for (int j = i + 1; j < existingNodes.size(); ++j) {
                int node1 = existingNodes[i];
                int node2 = existingNodes[j];
                matrix[node1][node2] = matrix[node2][node1] = 1;
            }
        }
    }

    void clear_mat(){
        for(int i = 0; i < n_nodes; ++i){
            for(int j = 0; j < n_nodes; ++j) matrix[i][j] = matrix[j][i] = 0;
        }
    }

    void connect(vector<int> path){
        clear_mat();
        for (int i = 0; i < path.size()-1; ++i) {
            int node1 = path[i];
            int node2 = path[i+1];
            matrix[node1][node2] = matrix[node2][node1] = 1;
        }
    }
};

#endif //TAREA_01_GRAPH_H