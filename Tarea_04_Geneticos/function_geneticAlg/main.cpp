#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

// Fitness functions
typedef vector<bool> dna;
typedef float (*fitness_function)(const vector<int> &);
typedef vector<pair<int, int>> delimiter;

// general data

int n_individuals = 8;
int n_variables = 3;
int ggg = 0;
int min_or_max;
vector<dna> generation;
vector<dna> offspring;
vector<int> indices;
vector<float> fitness_score;

vector<vector<float>> statistics;

void save_data(const string &file_name) {
  ofstream stats_file(file_name);
  if (stats_file.is_open()) {
    for (const auto &row : statistics) {
      for (const auto &value : row) {
        stats_file << value << "\t"; // Usa '\t' para tabulaciones
      }
      stats_file << "\n"; // Nueva línea al final de cada fila
    }
    stats_file.close();
  } 
}

float func1(const vector<int> &nums){
	int x = nums[0], y = nums[1];
	return 2*(x+y) - (x*y);
}

float func2(const vector<int> &nums) {
  int x = nums[0], y = nums[1];
  return x * x + y * y;
}

float func3(const vector<int> &nums){
	int x = nums[0], y = nums[1], z = nums[2];
	return 2*(x+y+z) - (x*y) + z;
}
void print(dna d) {
  for (auto bit : d)
    printf("%d", bit ? 1 : 0);
  printf("\n");
}

int dna_to_num(dna &variable) {
  int r = 0;
  for (bool bit : variable) {
    r = (r << 1) | bit;
  }
  return r;
}

int count_bits(int n) {
  return n == 0 ? 0 : 1 + (int)floor(log2(n));
  ;
}

vector<int> get_bits_perV(const vector<pair<int, int>> &delimiter) {
  vector<int> bits_perV;
  for (auto &p : delimiter) {
    bits_perV.push_back(count_bits(p.second));
  }
  return bits_perV;
}

dna create_individual(delimiter domain, vector<int> bits_per_var) {
  // srand(time(nullptr));
  dna individual;
  for (int i = 0; i < n_variables; i++) {
    // printf("WHAT a)%d  b) %d\n", domain[i].first, domain[i].second);
    int ran_num =
        domain[i].first + rand() % (domain[i].second - domain[i].first + 1);
    // printf("RAN: %d\n", ran_num);
    int parsing = bits_per_var[i] - count_bits(ran_num);
    while (parsing--) {
      individual.push_back(0);
    }
    while (ran_num > 0) {
      individual.push_back((ran_num & 1) == 0 ? 0 : 1);
      ran_num >>= 1;
    }
  }

  return individual;
}

void print_individual(const dna &individual, const vector<int> bits_perV) {
  int ini = 0;
  dna placeHolder;
  vector<int> nums;

  for (auto x : bits_perV) {
    //	printf("X: %d\n", x);
    int offset = ini;
    placeHolder = dna(individual.begin() + ini, individual.begin() + ini + x);
    for (int i = ini; i < individual.size() && i < offset + x; i++, ini++) {
      // printf("%d", individual[i] == 1 ? 1 : 0);
    }
    // placeHolder = dna(individual.begin() + ini, individual.begin() + ini +
    // x);
    //  nums.push_back(dna_to_num(placeHolder));
    // printf("\t%d\n", dna_to_num(placeHolder));
  }
}

void print_data(vector<int> bVp) {
  printf("\nIndividuo\tValores\tResultado\tPreselección\tValor esperado\tValor "
         "actual\n");
  int ini = 0;
  // int displacement = 0;
  dna placeHolder;
  vector<int> nums;

  float sum = accumulate(fitness_score.begin(), fitness_score.end(), 0);

  for (int i = 0; i < generation.size(); i++) {
    // in bits
    ini = 0;
    for (auto x : bVp) {
      placeHolder =
          dna(generation[i].begin() + ini, generation[i].begin() + ini + x);
      // in bits
      for (auto bit : placeHolder)
        printf("%d", bit ? 1 : 0);
      nums.push_back(dna_to_num(placeHolder));
      ini += x;
      printf(" ");
    }
    printf(" ");
    // in numbers
    for (auto x : nums)
      printf("%s%d%c ", x < 10 ? "0" : "", x, x != nums.back() ? ',' : ' ');
    nums.clear();
    // result
    printf("\t");
    printf("%.2f\t", fitness_score[i]);

    // Preselection
    printf("%.2f\t", fitness_score[i] / sum);

    // media
    float media = fitness_score[i] / (sum / n_individuals);
    printf("\t\t%.2f\t", media);

    // Valor esperado
    float decimal = media - floor(media);
    printf("\t\t%.2f\n", decimal >= 0.5 ? ceil(media) : floor(media));
  }

  printf("\n\nSUM: %.2f\n", sum);
  printf("Average: %.2f\n", sum / n_individuals);
}

void fitness_evaluation(fitness_function func, vector<int> bVp) {
  int bTd;
  dna placeHolder;
  vector<int> vars;
  for (int i = 0; i < n_individuals; ++i) {
    vars.clear();
    int ini = 0;
    for (int j = 0; j < n_variables; ++j) {
      placeHolder = dna(generation[i].begin() + ini,
                        generation[i].begin() + ini + bVp[j]);
      ini += bVp[j];
      bTd = dna_to_num(placeHolder);
      vars.push_back(bTd);
    }
    fitness_score.push_back(func(vars));
  }

  print_data(bVp);
}

void mutation(int index, vector<int> bits_per_var) {
  int ini = 0, index_adn;
  for (int i = 0; i < bits_per_var.size(); ++i) {
    index_adn = ini + (rand() % bits_per_var[i]);
    generation[index][index_adn] = !generation[index][index_adn];
    ini += bits_per_var[i];
  }
}

bool check_repeated(dna &genoma) {
  auto it = find(generation.begin(), generation.end(), genoma);
  return (it != generation.end());
}

void init_first_generation(delimiter &domain, vector<int> bits_per_var) {
  for (int i = 0; i < n_individuals; ++i) {
    dna placeHolder = create_individual(domain, bits_per_var);
    while (check_repeated(placeHolder)) {
      placeHolder = create_individual(domain, bits_per_var);
    }
    generation.push_back(placeHolder);
  }
}

void update_stats(int index) {
  float average =
      (float)accumulate(fitness_score.begin(), fitness_score.end(), 0) /
      n_individuals;
  ;
  statistics[index][0] = average;

  vector<pair<int, float>> indexed_fitness;
  for (int i = 0; i < fitness_score.size(); ++i) {
    indexed_fitness.push_back(std::make_pair(i, fitness_score[i]));
  }
  if (min_or_max)
    sort(indexed_fitness.begin(), indexed_fitness.end(),
         [](const auto &lhs, const auto &rhs) {
           return lhs.second > rhs.second;
         });
  else
    sort(indexed_fitness.begin(), indexed_fitness.end(),
         [](const auto &lhs, const auto &rhs) {
           return lhs.second < rhs.second;
         });

  for (const auto &pair : indexed_fitness) {
    indices.push_back(pair.first);
  }

  // statistics.push_back({sum / n_individuals, *max_element(fitness_score.
  if (min_or_max)
    sort(fitness_score.begin(), fitness_score.end(), std::greater<float>());
  else
    sort(fitness_score.begin(), fitness_score.end());
  statistics[index][1] = fitness_score.front();
  // statistics[index][2] = fitness_score.back();

  printf("STATS FOR GEN %d -- avg: %.2f, best: %.2f\n", index + 1,
         statistics[index][0], statistics[index][1]);
}

void gen_offspring(fitness_function func, vector<int> bits_per_v) {
  int i, j, l, cross_point, ini = 0;
  dna placeHolder;
  int parent_1 = indices[0];
  int parent_2 = indices[1];
  offspring.clear();
  offspring.assign(n_individuals - 2, dna());
  for (i = 0; i < n_individuals - 2; i += 2) {
    ini = 0;
    for (j = 0; j < n_variables; ++j) {
      cross_point = rand() % (bits_per_v[j]);

      // printf("ini: %d -cross_point: %d -bvp:%d\n", ini, cross_point,
      //      bits_per_v[j]);
      for (l = ini; l < cross_point + ini; ++l) {
        offspring[i].push_back(generation[parent_1][l]);
        offspring[i + 1].push_back(generation[parent_2][l]);
      }
      for (l = ini + cross_point; l < bits_per_v[j] + ini; ++l) {
        offspring[i].push_back(generation[parent_2][l]);
        offspring[i + 1].push_back(generation[parent_1][l]);
      }
      ini += bits_per_v[j];
    }
  }

  // copying and muting, both parents stay
  swap(generation[0], generation[parent_1]);
  swap(generation[1], generation[parent_2]);

  for (i = 2; i < n_individuals; ++i) {
    // printf("%d - el otro: %d\n", i, i - 2);
    generation[i] = offspring[i - 2];
    if (rand() % 100 < 95)
      mutation(i, bits_per_v);
  }
  // printf("Gets here? 2\n");
}

void genetic_algorithm(fitness_function func, int n_gens, delimiter &domain) {
  int i;
  // printf("Hola?\n");
  vector<int> bVp = get_bits_perV(domain);
  init_first_generation(domain, bVp);
  statistics.assign(n_gens, vector<float>(2));

  for (i = 0; i < n_gens; ++i) {
    ggg++;
    printf("----------GENERATION %d----------\n", i + 1);
    fitness_evaluation(func, bVp);
    // printf("...\n");
    //  getting stats
    update_stats(i);
    gen_offspring(func, bVp);
    indices.clear();
    fitness_score.clear();
  }
}


int main(int argc, char *argv[]) {
  srand(time(nullptr));

  delimiter domain = {{0, 15}, {16, 31}, {0,15}};
  int n_generations = 100;
  min_or_max = 0; // 0 for minimizing, 1 for maximizing a function
  genetic_algorithm(func3, n_generations, domain);
  save_data("stats.txt");
  //vector<vector<double>> data = load_data("stats.txt");
  //plotting(data);
}