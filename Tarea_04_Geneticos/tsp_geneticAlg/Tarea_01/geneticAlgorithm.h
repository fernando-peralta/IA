#include "graph.h"
#include "graficos.h"
#include <fstream>

int graphSize = 15;
Graph graph(graphSize, graphSize);

vector<Line> lines;
vector<Circle> circles;
vector<Line> foundPath;




void printNodes(){
    vector<int> nodes = graph.getNodes();
    for(auto& n : nodes){
        printf("%d ", n);
    }
    printf("\n");
}

typedef vector<int> road;
typedef int (*fitness_function)(const vector<int> &);
typedef vector<pair<int, int>> delimiter;

int n_individuals = 20;
int initiated = 0;

vector<road> generation;
vector<road> offspring;
vector<int> indices;
vector<float> fitness_score;
vector<vector<float>> statistics;


void save_data(const string &file_name) {
  ofstream stats_file(file_name);
  if (stats_file.is_open()) {
    for (const auto &row : statistics) {
      for (const auto &value : row) {
        stats_file << value << "\t"; 
      }
      stats_file << "\n"; 
    }
    stats_file.close();
  } 
}



void print_gen(road& gen){
    for(int i = 0; i < gen.size(); ++i){
        printf("%d%s", gen[i], (i==gen.size()-1) ? "" : "-");
    }
    printf("\n");
}


road create_individual() {
    road point = graph.getNodes();
    random_shuffle(point.begin(), point.end());
    point.push_back(point.front()); //vuelve al inicio
    //print_gen(point);
    return point;
}

bool check_repeated(road &genoma) {
  auto it = find(generation.begin(), generation.end(), genoma);
  return (it != generation.end());
}

void init_first_generation() {
  for (int i = 0; i < n_individuals; ++i) {
    road placeHolder = create_individual();
    while (check_repeated(placeHolder)) {
      placeHolder = create_individual();
    }
    generation.push_back(placeHolder);
  }

}




void print_data() {
  printf("\nIndividuo\t\t\tResultado\tPreselección\tValor esperado\tValor "
         "actual\n");
  int ini = 0;
  int n;
  // int displacement = 0;
  vector<int> nums;

  float sum = accumulate(fitness_score.begin(), fitness_score.end(), 0);

  for (int i = 0; i < generation.size(); i++) {

    for(int j = 0; j < generation[i].size(); ++j){
        n = generation[i][j];
        printf("%s%d%s", (n < 10) ? " " : "", n, (j == generation[i].size()-1) ? "" : "-");
    }
    //fitness
    printf("\t");
    printf("%.2f\t\t", fitness_score[i]);

    // Preselection
    printf("%.2f\t", fitness_score[i] / sum);

    // media
    float media = fitness_score[i] / (sum / n_individuals);
    printf("\t%.2f\t", media);

    // Valor esperado
    float decimal = media - floor(media);
    printf("\t\t%.2f\n", decimal >= 0.5 ? ceil(media) : floor(media));
  }

  printf("\n\nSUM: %.2f\n", sum);
  printf("Average: %.2f\n", sum / n_individuals);
}

void fitness_evaluation() {
    float total_distance;
    for(int i = 0; i < n_individuals; ++i){
        total_distance = 0;
        for(int j = 0; j < generation[i].size()-1; ++j){
            total_distance += graph.distance(generation[i][j], generation[i][j+1]);
        }
        fitness_score.push_back(total_distance);
    }
    //print_data();
}


void update_stats(int index) {
  float average = (float)accumulate(fitness_score.begin(), fitness_score.end(), 0) / n_individuals;
  statistics[index][0] = average;

  vector<pair<int, float>> indexed_fitness;
  for (int i = 0; i < fitness_score.size(); ++i) {
    indexed_fitness.push_back(make_pair(i, fitness_score[i]));
  }
  sort(
      indexed_fitness.begin(), indexed_fitness.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });

  for (const auto &pair : indexed_fitness) {
    //printf("%d ", pair.first);
    indices.push_back(pair.first);
  }
  //printf("\n");
  sort(fitness_score.begin(), fitness_score.end());
  //printf("BEST: %.2f -- WORST: %.2f\n", fitness_score.front(), fitness_score.back());
  statistics[index][1] = fitness_score.front();
  // statistics[index][2] = fitness_score.back();

  // printf("STATS FOR GEN %d -- avg: %.2f, best: %.2f, min: %.2f\n", index+1,
  //        statistics[index][0], statistics[index][1], statistics[index][2]);
  //printf("STATS FOR GEN %d -- avg: %.2f, best: %.2f\n", index + 1, statistics[index][0], statistics[index][1]);
}



road crossOver(road parent1, road parent2) {
    int size = parent1.size();
    road child;
    int start = rand() % size;
    int end = rand() % size;
    int i;
    if (start > end) swap(start, end);
    for (i = start; i <= end; ++i) {
        child.push_back(parent1[i]);
    }

    //printf("\nstart: %d, end:%d\n", start, end);

    if(abs(end-start) == (size-1)) {
        child.push_back(child.front());
        return child;
    }

    for(int j = 0; j < size; ++j){
        if(find(child.begin(), child.end(), parent2[j]) == child.end()){
            child.push_back(parent2[j]);
        }
    }
    //if(end != size-1) 
    child.push_back(child.front());

    return child;
}


void mutation(int index){
    
    int pos_1 = 1 + rand() % (generation[index].size() - 2); 
    int pos_2 = 1 + rand() % (generation[index].size() - 2); 
    //printf("\nMUTING --- p1:%d  -- p2:%d\n", pos_1, pos_2);
    swap(generation[index][pos_1], generation[index][pos_2]);
}   

void gen_offspring() {
  int i, j, l, cross_point, ini = 0;
  int parent_1 = indices[0];
  int parent_2 = indices[1];

  offspring.clear();

  for (i = 0; i < n_individuals - 2; i++) {
    offspring.push_back(crossOver(generation[parent_1], generation[parent_2]));
  }

  
  swap(generation[0], generation[parent_1]);
  swap(generation[1], generation[parent_2]);

  for (i = 2; i < n_individuals; ++i) {
    generation[i] = offspring[i - 2];
    if (rand() % 100 < 95) mutation(i);
  }
}


void genetic_algorithm(int n_gens = 500){
  generation.clear();
  offspring.clear();
  indices.clear();
  fitness_score.clear();
  statistics.clear();
    init_first_generation();
    statistics.assign(n_gens, vector<float>(2));
    int i = 0;
    initiated = 1;
    for( i = 0; i < n_gens; ++i){
        printf("----------GENERATION %d----------\n", i + 1);
        fitness_evaluation();
        // printf("...\n");
        //  getting stats
        update_stats(i);
        gen_offspring();
        indices.clear();
        fitness_score.clear();
    }
    //update_stats(i);
    graph.connect(generation[0]);
    printf("BEST SOLUTION: ");
    print_gen(generation[0]);
    //printf("Stats size?? %d\n", statistics.size());
    save_data("stats.txt");
    
}
