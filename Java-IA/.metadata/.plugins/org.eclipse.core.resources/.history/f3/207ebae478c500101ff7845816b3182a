package Ejercicios;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Ejercicio1 {

	public static void main(String[] args) {
		// Analisis exploratiorio iris weka
		try {
			Instances iris = DataSource.read("wine.arff");// CARGAMOS EL DATASET (LECTURA)
			int nInstances = iris.numInstances();// OBTENEMOS LA CANTIDAD DE REGISTROS
			int nAttributes = iris.numAttributes();// CANTIDAD DE COLUMNAS (CONTANDO CON EL TARGET)
			System.out.println("Cantidad de muestras: " + nInstances);
			System.out.println("Cantidad de atributos: " + (nAttributes - 1));// RESTAS EL TARGET
			System.out.println("---Atributos---");
			for (int i = 0; i < iris.numAttributes(); i++) {
				System.out.println(iris.attribute(i).name());
			}
			System.out.println("---Instancias---");
			for (int i = 0; i < iris.numInstances(); i++) {
				System.out.println(iris.instance(i));
			}
		} catch (Exception e) {
		}
	}

}
