#include <QApplication>
#include <QMainWindow>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
#include <iostream>
#include <string>
#include <limits.h>

using std::cin;
using std::cout;
using std::endl;

int **board;
QPushButton ***buttons;
int canPlay = 0;

typedef struct node node;

struct node {
    float val;
    int x, y;
    int alpha, beta;
    int nnext;
    node **next;
    node *parent;
    int **boardState;
    int size;
};

void
printBoard(int **board, int n)
{
    printf(" ");
    for (int j = 0; j < n; j++) {
        printf(" %c ", j + 'a');
    }
    for (int i = 0; i < n; i++) {
        printf("\n%c", i + 'A');
        for (int j = 0; j < n; j++) {
            printf("|");
            switch (board[i][j]) {
                case 1:
                    printf("X");
                    break;
                case -1:
                    printf("O");
                    break;
                default:
                    printf(" ");
                    break;
            }
            printf("|");
        }
        printf("\n");
    }
}

float
evalFunction(int **board, int n, int currentEval)
{
    float accH = 0;
    float accV = 0;
    for (int j = 0; j < n; j++) {
        float paccH = 0;
        float paccV = 0;
        float scoreH = 2;
        float scoreV = 2;
        for (int i = 0; i < n; i++) {
            if (board[j][i] == currentEval)
                paccH += scoreH;
            else if (board[j][i] == (currentEval * -1))
                paccH = scoreH = -1;

            if (board[i][j] == currentEval)
                paccV += scoreV;
            else if (board[i][j] == (currentEval * -1))
                paccV = scoreV = -1;
        }
        accH += paccH;
        accV += paccV;
    }

    accH = (accH == 0 ? 0 : accH / n);
    accV = (accV == 0 ? 0 : accV / n);

    float accD = 0;
    float scoreD = 2;

    float accDD = 0;
    float scoreDD = 2;
    for (int i = 0; i < n; i++) {
        if (board[i][i] == currentEval)
            accD += scoreD;
        else if (board[i][i] == (currentEval * -1))
            accD = scoreD = -1;

        if (board[i][n - i - 1] == currentEval)
            accDD += scoreDD;
        else if (board[i][n - i - 1] == (currentEval * -1))
            accDD = scoreDD = -1;

    }

    // printf("H: %f, V: %f, D: %f\n", accH, accV, accD);
    float winP = (accH == 0 ? 0 : accH / n) + (accV == 0 ? 0 : accV / n) + (accD == 0 ? 0 : accD / n) + (accDD == 0 ? 0 : accDD / n);
    return winP == 0 ? 0 : winP / 3;
}

node*
initNode(int **board, int n)
{
    node *root = (node*)malloc(sizeof(node));
    root->alpha = INT_MIN;
    root->beta = INT_MIN;
    root->parent = NULL;
    root->next = NULL;
    root->size = n;

    root->boardState = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        root->boardState[i] = (int*)malloc(sizeof(int) * n);
        for (int j = 0; j < n; j++) {
            root->boardState[i][j] = board[i][j];
        }
    }
    return root;
}

void
genTree(node *root, int availableMoves, int depth, int currentPlayer)
{
    if (depth == 0) {
        root->val = evalFunction(root->boardState, root->size, currentPlayer) - evalFunction(root->boardState, root->size, currentPlayer * -1);
        // printBoard(root->boardState, root->size);
        // printf("%f\n", evalFunction(root->boardState, root->size, currentPlayer) - evalFunction(root->boardState, root->size, currentPlayer * -1));
    }

    root->next = (node**)malloc(sizeof(node*) * availableMoves);

    int count = 0;
    for (int i = 0; i < root->size; i++) {
        for (int j = 0; j < root->size; j++) {
            //printf("i: %d, j: %d\n", i, j);
            if (root->boardState[i][j] == 0) {
                root->next[count] = initNode(root->boardState, root->size);
                root->next[count]->boardState[i][j] = currentPlayer;
                root->next[count]->x = j;
                root->next[count]->y = i;
                genTree(root->next[count], availableMoves - 1, depth - 1, currentPlayer * -1);
                count++;
            }
        }
    }

    root->nnext = count;
}

float
minMax(node *root, int isMaximizing, int depth, int *x, int *y)
{
    if (depth == 0) {
        return root->val;
    }

    float c = isMaximizing == 1 ? INT_MIN : INT_MAX;
    for (int i = 0; i < root->nnext; i++) {
        float t = minMax(root->next[i], ~isMaximizing, depth - 1, x, y);

        if (isMaximizing == 1) {
            if (t > c) {
                c = t;
                *x = root->next[i]->x;
                *y = root->next[i]->y;
            }
        } else {
            if (t < c) {
                c = t;
                *x = root->next[i]->x;
                *y = root->next[i]->y;
            }
        }
    }

    return c;
}

int
gameStatus(int x, int y, int **board, int n)
{
    int player = board[y][x];
    int accH = 0;
    int accV = 0;
    int accD = 0;
    int accDD = 0;
    for (int i = 0; i < n; i++) {
        accH += board[y][i];
        accV += board[i][x];
        accD += board[i][i];
        accDD += board[i][n - i - 1];
    }

    int winScore = player * n;
    if (accDD == winScore || accD == winScore || accH == winScore || accV == winScore)
        return 1;
    return 0;
}

int
getUserInput(std::string message) 
{
    QDialog dialog;
    dialog.setWindowTitle("Input Dialog");

    QVBoxLayout layout(&dialog);
    QLabel label(QString::fromStdString(message)); 
    QLineEdit input;
    QPushButton submit("Ingresar");

    layout.addWidget(&label);
    layout.addWidget(&input);
    layout.addWidget(&submit);

    int result = 0;

    QObject::connect(&submit, &QPushButton::clicked, [&]() {
        QString userInput = input.text();
        bool ok;
        result = userInput.toInt(&ok);
        if (!ok) {
            QMessageBox::critical(&dialog, "Error", "Invalid input. Please enter a valid integer.");
            return;
        }
        dialog.close();
    });

    dialog.exec();
    return result;
}

int 
getYesNoDialog(const QString& title, const QString& message)
{
  QMessageBox dialog;
  dialog.setWindowTitle(title);
  dialog.setText(message);
  dialog.setStandardButtons(QMessageBox::Yes | QMessageBox::No);

  int result = dialog.exec();

  return (result == QMessageBox::Yes) ? 1 : 0;
}

void
showMessageDialog(const QString& message)
{
    QMessageBox msgBox;
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(message);
    msgBox.setWindowTitle("Message");
    
    msgBox.addButton(QMessageBox::Ok);
    msgBox.exec();
}

int 
main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    int n = getUserInput("Ingrese el tamaño del tablero: ");
    int d = getUserInput("Ingrese la profundidad del algoritmo: ");
    canPlay = getYesNoDialog("", "Desea usted comenzar? ");

    board = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        board[i] = (int*)malloc(sizeof(int) * n);
        for (int j = 0; j < n; j++) {
            board[i][j] = 0;       
        }
    }

    QWidget mainWindow;
    mainWindow.setWindowTitle("Tic-Tac-Toe");

    QGridLayout *layout = new QGridLayout(&mainWindow);

    //QPushButton *buttons[n][n];
    buttons = new QPushButton**[n];
    for (int row = 0; row < n; ++row) {
        buttons[row] = new QPushButton*[n];
        for (int col = 0; col < n; ++col) {
            buttons[row][col] = new QPushButton("", &mainWindow);
            QFont font("Hack", 50, 1);
            buttons[row][col]->setFont(font);
            buttons[row][col]->setFixedSize(100, 100);
            layout->addWidget(buttons[row][col], row, col);

            QObject::connect(buttons[row][col], &QPushButton::clicked, [row, col, d, n, &app]() {
                if (board[row][col] == 0 && canPlay == 1) {
                    board[row][col] = 1;
                    buttons[row][col]->setText("X");
                    buttons[row][col]->setDisabled(true);
                    // currentPlayer = currentPlayer == 1 ? -1 : 1;
                    
                    if (gameStatus(col, row, board, n)) {
                        showMessageDialog("Usted GANÓ!");
                        app.quit();
                    }
                    
                    canPlay = 0;

                    int x = -1;
                    int y = -1;
                    node root = initNode(board, n);
                    genTree(root, availableMoves--, d, -1);
                    minMax(root, 1, d, &x, &y);
                    board[y][x] = -1;
                    buttons[y][x]->setText("O");
                    buttons[y][x]->setDisabled(true);
                    if (gameStatus(x, y, board, n)) {
                        showMessageDialog("Usted PERDIÓ!");
                        app.quit();
                    }

                    canPlay = 1;
                    printBoard(board, n);
                }
            });
        }
    }

    mainWindow.setLayout(layout);

    mainWindow.show();
    return app.exec();
}
