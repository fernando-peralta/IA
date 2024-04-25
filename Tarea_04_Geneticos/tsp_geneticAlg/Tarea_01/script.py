import matplotlib.pyplot as plt

def load_data(file_name):
        d = []
        with open(file_name, 'r') as file:
                for line in file:
                        row = [float(val) for val in line.split()]
                        d.append(row)
        return d


scripts = ["Promedio", "Menor distancia"]


def plotting(data):
        plt.figure(figsize=(8, 6))  
        for column in range(len(data[0])):
                col_values = [fila[column] for fila in data]
                plt.plot(col_values, label=scripts[column])  
        plt.xlabel('Generación')
        plt.ylabel('Funcion de aptitud')
        plt.title('Estadísticas sobre las generaciones para la función')
        plt.legend()  
        plt.grid(True)  
        plt.show()

def main():
        data = load_data(r"C:\Users\asus\Downloads\tsp_geneticAlg\Tarea_01")
        plotting(data)

if __name__ == "__main__":
        main()